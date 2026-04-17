#define TRIP_BACKWARD_VERSION 2026041501
#include "trip.h"

// ============================================================
//  backward()
//
//  The mirror of forward(): walk the same layers in reverse,
//  computing how much each weight contributed to the loss.
//
//  The flow, from output back to input:
//
//  1. LOSS GRADIENT — compute dlogits from the cross-entropy
//     loss between predicted probabilities and the target
//     token.
//
//  2. CLASSIFIER GRADIENT — backprop through the final
//     projection (logits = stream · classifier_weights).
//
//  3. FINAL NORM GRADIENT — backprop through the final
//     normalization.
//
//  4. Walk the layers in REVERSE (from layer N-1 down to 0).
//     At each layer, the mirror of forward:
//     f) RESIDUAL — split the gradient into FFN path +
//        skip path
//     e) FFN BACKWARD — backprop through down-project,
//        non-linearity, gate, up-project; accumulate
//        weight gradients
//     d) POST-ATTENTION NORM BACKWARD
//     c) RESIDUAL — split the gradient into attention path
//        + skip path
//     b) ATTENTION BACKWARD — backprop through output
//        projection, attention scores, Q/K/V projections;
//        accumulate weight gradients
//     a) PRE-ATTENTION NORM BACKWARD
//
//  5. EMBEDDING GRADIENT — scatter the gradient back to the
//     embedding table entries that were looked up.
//
//  After backward() completes, all weight gradients are
//  accumulated in model->grads. The optimizer (AdamW) then
//  uses them to update the weights — see model_update().
//
//  For an excellent introduction to backpropagation, see
//  Andrej Karpathy's lecture:
//  https://www.youtube.com/watch?v=i94OvYb6noo
//  TRiP would never have existed without his work.
// ============================================================

int backward(Model * model, int attention_type, int ** intok, size_t B, size_t T){

	if(runtime_actions & (1<<SIGUSR2))	return -1;

	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t dim_stream = (size_t)(model->config.dim_stream);
	//size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
	size_t max_tokens =  T  ;
	size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
	size_t vocab_size = (size_t)(model->config.vocab_size);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_queries  = (size_t)(model->config.n_queries);
	size_t n_keys     = (size_t)(model->config.n_keys);


float norm;

mylog(LOG_DEBUG,"Entering backward()!");

	//let's backprop into the loss calculation (crossentropy) and, together, the softmax to extrapolate the output probabilities over all the vocabulary entries
	crossentropy_softmax_backward(model->grads.dlogits, model->fm.logits, intok, model, B, T);


#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"norm dlogits = %.5f", tensorgrad_norm_squared(NULL, model->grads.dlogits, vocab_size * B * T));
#endif


mylog(LOG_VERBOSE_DEBUG,"let's backprop into the logits calculation");

	if(runtime_actions & (1<<SIGUSR2))	return -1;

	//let's backprop into the logits calculation
	matmulf_nt_backward(
		model->grads.dlogits_classifier, model->grads.dnorm_final_stream,
		model->w.logits_classifier, model->fm.norm_final_stream, dim_stream,vocab_size, dim_stream,(B*T), model->grads.dlogits,
		B, T
	);


#ifdef TRIP_DEBUG
wdebug(model->grads.dembeddings, WTYPE_FLOAT32, 5, "model->grads.dembeddings (as classifier)", -1, 0);
#endif

	
#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"norm dnorm_final_stream = %.7f", tensorgrad_norm_squared(NULL, model->grads.dnorm_final_stream, dim_stream * B * T));
mylog(LOG_VERBOSE_DEBUG,"norm norm_final_stream = %.7f", tensorgrad_norm_squared(NULL, model->fm.norm_final_stream, dim_stream * B * T));
mylog(LOG_VERBOSE_DEBUG,"norm dlogits_classifier = %.7f", tensorgrad_norm_squared(NULL, model->grads.dlogits_classifier, dim_stream * vocab_size));
#endif


mylog(LOG_VERBOSE_DEBUG,"let's backprop into the FINAL normalization");

	//let's backprop into the FINAL normalization (output layer, right after the last Attention+FFN layer)
	switch(model->config.norm_cfg[2]){

	    case NORM_NONE:	
		//just backpropagate from output to input
		sum_vectors_backward(
			model->grads.dresidualstream_after_ffn[n_layers - 1], model->grads.dnorm_final_stream,
			model->grads.dnorm_final_stream,
			(dim_stream * B * T)
		);	
		break;	//do nothing
					
	    case NORM_LAYERNORM: 	//LAYERNORM
		layernorm_backward(
			model->grads.dresidualstream_after_ffn[n_layers - 1],  model->grads.dnorm_final_w,  model->grads.dnorm_final_b,
			model->grads.dnorm_final_stream,  model->fm.residualstream_after_ffn[n_layers - 1],  model->w.norm_final_w,  model->fm.norm_final_mean,  model->fm.norm_final_rstd,
			B, T, dim_stream
		);
		break;

	    case NORM_RMSNORM:	//RMSNORM
		rmsnorm_backward(
			model->grads.dresidualstream_after_ffn[n_layers - 1],  model->grads.dnorm_final_w,
			model->grads.dnorm_final_stream,  model->fm.residualstream_after_ffn[n_layers - 1],  model->w.norm_final_w,  model->fm.norm_final_rrms,
			model, B, T, dim_stream
		);

  		break;

	    default: 
		mylog(LOG_ERROR, "Backward function for FINAL normalization %02X not implemented. Exiting...", model->config.norm_cfg[2]);
		exit(-1);
		break;
	}


#ifdef TRIP_DEBUG
wdebug(model->fm.norm_final_rrms, WTYPE_FLOAT32, (B*T), "model->fm.norm_final_rrms[B][T]", -1, 1);
wdebug(model->grads.dresidualstream_after_ffn[n_layers - 1], WTYPE_FLOAT32, 5, "model->grads.dresidualstream_after_ffn[n_layers - 1]", -1, 0);
#endif


	if(runtime_actions & (1<<SIGUSR2))	return -1;


mylog(LOG_VERBOSE_DEBUG,"let's loop over the layers!");


	//let's loop over the layers, starting from the last one, and backprop to the top
	for(ssize_t layer = (n_layers - 1); layer >= 0 ; layer --){


mylog(LOG_DEBUG,"Entering layer #%d!",layer);

mylog(LOG_VERBOSE_DEBUG,"let's backprop into the residual connection after the post-FFN linear layer");

		//let's backprop into the residual connection after the post-FFN linear layer
		sum_vectors_backward(
			model->grads.dresidualstream_after_attention[layer], model->grads.dffn_final_stream[layer],
			model->grads.dresidualstream_after_ffn[layer],
			(dim_stream * B * T)
		);



#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dffn_final_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dffn_final_stream[layer], dim_stream*B*T));
#endif


mylog(LOG_VERBOSE_DEBUG,"let's backprop into the linear layer AFTER the Feed Forward Network");

		//let's backprop into the linear layer AFTER the Feed Forward Network
		if(model->config.bias_cfg[1] == BIAS_ON){

			//WARNING:	I am just propagating(reusing) the output gradient vector to the input gradient vector
			//WARNING #2:	I cannot use (dim_stream*B*T) because dpost_ffn_b has size dim_stream

			//#pragma omp parallel for collapse(2)	//NO! there would be a race on model->grads.dpost_ffn_b when accumulating gradients
			for(size_t b = 0; b < B; b++){
				for(size_t t = 0; t < T; t++){

					size_t bto = ((b*T)+t)*dim_stream*sizeof(float);	//offset

					sum_vectors_backward(
						&model->grads.dffn_final_stream[layer][bto], model->grads.dpost_ffn_b[layer],
						&model->grads.dffn_final_stream[layer][bto],
						dim_stream
					);
				}
			}
		}


#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dffn_final_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dffn_final_stream[layer], dim_stream*B*T));
#endif


mylog(LOG_VERBOSE_DEBUG,"let's backprop into the linear layer (matmul) AFTER the Feed Forward Network");
		matmulf_nt_backward(
			model->grads.dpost_ffn_w[layer], model->grads.dffn_out_stream[layer],
			model->w.post_ffn_w[layer], model->fm.ffn_out_stream[layer], hidden_dim,dim_stream, hidden_dim,(B*T), model->grads.dffn_final_stream[layer],
			B, T
		);


#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dffn_out_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dffn_out_stream[layer], dim_stream*B*T));
mylog(LOG_VERBOSE_DEBUG,"norm dpost_ffn_b = %.7f", tensorgrad_norm_squared(NULL, model->grads.dpost_ffn_b[layer], dim_stream));
mylog(LOG_VERBOSE_DEBUG,"norm dpost_ffn_w = %.7f", tensorgrad_norm_squared(NULL, model->grads.dpost_ffn_w[layer], dim_stream*hidden_dim));
#endif



mylog(LOG_VERBOSE_DEBUG,"let's backprop into the Feed Forward Network itself");

		//let's backprop into the Feed Forward Network itself

		//this is the management of a special case: gated non-linearity, like LLAMA2 SILU, which requires additional multipliers
		int ffn_gating;

		if((model->config.ffn_nl_type[1] == GATE_ON)  ||  (model->config.ffn_nl_type[0] == NL_SILU_LLAMA))	ffn_gating = 1;
		else													ffn_gating = 0;
			

		#pragma omp parallel for collapse(2)
		for(size_t b = 0; b < B; b++){
		    for(size_t t = 0; t < T; t++){
			
			size_t bto = ((b*T)+t)*hidden_dim*sizeof(float);	//offset
			
			ffn_io_backward(
			        &model->grads.dffn_in_stream[layer][bto], ((ffn_gating==1) ? &model->grads.dffn_aux_stream[layer][bto] : NULL),
		        	&model->grads.dffn_out_stream[layer][bto], &model->fm.ffn_in_stream[layer][bto], ((ffn_gating==1) ? &model->fm.ffn_aux_stream[layer][bto] : NULL),
			        model->config.ffn_nl_type[0], hidden_dim
			);
		    }
		}


#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dffn_in_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dffn_in_stream[layer], dim_stream*B*T));
#endif


		if(ffn_gating == 1){
			
mylog(LOG_VERBOSE_DEBUG,"let's backprop into the calculation of the gating vector for the FFN");
	
			//matmulf_nt(model->w.pre_ffn_w2[layer], bstream, dim_stream,hidden_dim, dim_stream,(B*T), ffn_aux_stream[layer]);

			matmulf_nt_backward(
				model->grads.dpre_ffn_w2[layer], model->grads.dffn_in_stream[layer],
				model->w.pre_ffn_w2[layer], model->fm.ffn_in_stream[layer], dim_stream,hidden_dim, dim_stream,(B*T), model->grads.dffn_aux_stream[layer],
				B, T
			);

		}

#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dffn_in_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dffn_in_stream[layer], dim_stream*B*T));
mylog(LOG_VERBOSE_DEBUG,"norm  model->grads.dpre_ffn_w2[layer] = %.7f", tensorgrad_norm_squared(NULL, model->grads.dpre_ffn_w2[layer], dim_stream*hidden_dim));
#endif


		
mylog(LOG_VERBOSE_DEBUG,"let's backprop into the linear layer BEFORE the Feed Forward Network");

		//let's backprop into the linear layer BEFORE the Feed Forward Network
		if(model->config.bias_cfg[0] == BIAS_ON){

			//WARNING:	I am just propagating(reusing) the output gradient vector to the input gradient vector
			//WARNING #2:	I cannot use (dim_stream*B*T) because dpre_ffn_b has size dim_stream

			//#pragma omp parallel for collapse(2)	//NO! there would be a race on model->grads.dpre_ffn_b when accumulating gradients
			for(size_t b = 0; b < B; b++){
				for(size_t t = 0; t < T; t++){

					size_t bto = ((b*T)+t)*dim_stream*sizeof(float);	//offset

					sum_vectors_backward(
						&model->grads.dffn_in_stream[layer][bto], model->grads.dpre_ffn_b[layer],
						&model->grads.dffn_in_stream[layer][bto],
						dim_stream
					);
				}
			}
		}

#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dffn_in_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dffn_in_stream[layer], dim_stream*B*T));
#endif
		
mylog(LOG_VERBOSE_DEBUG,"let's backprop into the linear layer (matmul) BEFORE the Feed Forward Network");

		matmulf_nt_backward(
			model->grads.dpre_ffn_w[layer], model->grads.dnorm_post_stream[layer],
			model->w.pre_ffn_w[layer], model->fm.norm_post_stream[layer], dim_stream,hidden_dim, dim_stream,(B*T), model->grads.dffn_in_stream[layer],
			B, T
		);

#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"norm dpre_ffn_b = %.7f", tensorgrad_norm_squared(NULL, model->grads.dpre_ffn_b[layer], dim_stream));
mylog(LOG_VERBOSE_DEBUG,"norm dpre_ffn_w = %.7f", tensorgrad_norm_squared(NULL, model->grads.dpre_ffn_w[layer], dim_stream*hidden_dim));
#endif


#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dnorm_post_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dnorm_post_stream[layer], dim_stream*B*T));
#endif


mylog(LOG_VERBOSE_DEBUG,"let's backprop into the POST ATTENTION normalization");

		//let's backprop into the POST ATTENTION normalization (right after the output projection(+bias) from the attention layer)
		switch(model->config.norm_cfg[1]){
	
		    case NORM_NONE:	
			//just backpropagate from output to input
			sum_vectors_backward(
				model->grads.dresidualstream_after_attention[layer], model->grads.dnorm_post_stream[layer],
				model->grads.dnorm_post_stream[layer],
				(dim_stream * B * T)
			);	
			break;	//do nothing
						
		    case NORM_LAYERNORM: 	//LAYERNORM
			layernorm_backward(
				model->grads.dresidualstream_after_attention[layer],  model->grads.dnorm_post_w[layer],  model->grads.dnorm_post_b[layer],
				model->grads.dnorm_post_stream[layer],  model->fm.residualstream_after_attention[layer],  model->w.norm_post_w[layer],  model->fm.norm_post_mean[layer],  model->fm.norm_post_rstd[layer],
				B, T, dim_stream
			);
			break;
	
		    case NORM_RMSNORM:	//RMSNORM
			rmsnorm_backward(
				model->grads.dresidualstream_after_attention[layer],  model->grads.dnorm_post_w[layer],
				model->grads.dnorm_post_stream[layer],  model->fm.residualstream_after_attention[layer],  model->w.norm_post_w[layer],  model->fm.norm_post_rrms[layer],
				model, B, T, dim_stream
			);
	
	  		break;
	
		    default: 
			mylog(LOG_ERROR, "Backward function for POST-ATTENTION normalization %02X not implemented. Exiting...", model->config.norm_cfg[1]);
			exit(-1);
			break;
		}


#ifdef TRIP_DEBUG
wdebug(model->grads.dresidualstream_after_attention[layer], WTYPE_FLOAT32, 5, "model->grads.dresidualstream_after_attention[layer]", -1, 0);
#endif

#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dresidualstream_after_attention[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dresidualstream_after_attention[layer], dim_stream*B*T));
#endif


#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"norm  model->grads.dnorm_post_w[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dnorm_post_w[layer], dim_stream));
#endif

mylog(LOG_VERBOSE_DEBUG,"let's backprop into the residual connection after the ATTENTION layer");

		//let's backprop into the residual connection after the ATTENTION layer
		sum_vectors_backward(
			model->grads.dresidualstream_layerstart[layer], model->grads.dattentionlayer_out_stream[layer],
			model->grads.dresidualstream_after_attention[layer],
			(dim_stream * B * T)
		);



#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dresidualstream_after_attention[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dresidualstream_after_attention[layer], dim_stream*B*T));
#endif

#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dattentionlayer_out_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dattentionlayer_out_stream[layer], dim_stream*B*T));
#endif


	
mylog(LOG_VERBOSE_DEBUG,"let's backprop into the output projection (+bias) from the attention layer");

		//let's backprop into the output projection (+bias) from the attention layer 
		if(model->config.bias_cfg[3] == BIAS_ON){

			//WARNING:	I am just propagating(reusing) the output gradient vector to the input gradient vector
			//WARNING #2:	I cannot use (dim_stream*B*T) because grads.dob has size dim_stream

			//#pragma omp parallel for collapse(2)	//NO! there would be a race on model->grads.dob when accumulating gradients
			for(size_t b = 0; b < B; b++){
				for(size_t t = 0; t < T; t++){

					size_t bto = ((b*T)+t)*dim_stream*sizeof(float);	//offset

					sum_vectors_backward(
						&model->grads.dattentionlayer_out_stream[layer][bto], model->grads.dob[layer],
						&model->grads.dattentionlayer_out_stream[layer][bto],
						dim_stream
					);
				}
			}
		}

		
#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dattentionlayer_out_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dattentionlayer_out_stream[layer], dim_stream*B*T));
#endif

		matmulf_nt_backward(
			model->grads.dom[layer], model->grads.dheads_output[layer],
			model->w.om[layer], model->fm.heads_output[layer], dim_stream,dim_stream, dim_stream,(B*T), model->grads.dattentionlayer_out_stream[layer],
			B, T
		);

#ifdef TRIP_DEBUG
wdebug(model->grads.dheads_output[layer], WTYPE_FLOAT32, 5, "model->grads.dheads_output[layer]", -1, 0);
wdebug(model->grads.dheads_output[layer] + ((768 + 95)*sizeof(float)), WTYPE_FLOAT32, 5, "model->grads.dheads_output[layer] + ((768 + 95)*sizeof(float))", -1, 0);
#endif

#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dheads_output[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dheads_output[layer], dim_stream*B*T));
#endif


mylog(LOG_VERBOSE_DEBUG,"let's backprop into the attention layer");

		//and now, let's backprop into the attention layer

			//now we cycle over all the attention heads (same number of the queries, in multi-head, multi-query, and grouped-query attention schemes)

			size_t last_qhead  = -1;	//yes, it's -1 to unsigned; I just want to put a bad value in it
			size_t last_kvhead = -1;

			for(size_t head = 0; (head < n_queries); head++){

				//generalization for each possible case at the best of my knowledge:
				// - "multi-head"	(number of queries = number of keys&values)	
				// - "multi-query"	(number of keys&values = 1)
				// - "grouped-query"	(number of keys and values = sub-multiple for number of queries)

				size_t kv = ((head * n_keys)  /  n_queries);


mylog(LOG_VERBOSE_DEBUG,"(entering head %d/%d, kv %d)",head,n_queries,kv);

	size_t q_offset;


	q_offset  = 0;
	q_offset += head;			//currently, we are processing query #head (over n_queries in total)
	q_offset *= B;				//each query area in the cache is a sub-block, made up of "B" sequences
	//q_offset += b;			//currently, we are processing sequence #b (over B sequences in total)
	q_offset *= max_tokens;			//each sequence/query area in the cache is a sub-sub-block, made up of "sequence_maxtokens" positions
	//q_offset += pos;			//currently, we are processing position #pos (over sequence_maxtokens in total)
	q_offset *= dim_qkv;			//each pos/sequence/query in the cache is a query vector, made up of "dim_qkv" components
	//since the cache is actually "float *", we need to multiply by the size of float32
	q_offset *= sizeof(float);

	byte * q  = &model->fm.queries[layer][q_offset];
	byte * dq = &model->grads.dqueries[layer][q_offset];




	//NOW, to prepare the attention computation, both for keys and values we need to point to the cache for this layer/head
	//from sequence ZERO, pos ZERO, and NOT from the current position

	size_t kv_offset;

	kv_offset  = 0;
	kv_offset += kv;				//currently, we are processing key #kv (over n_keys in total)
	kv_offset *= B;
	//kv_offset += b;				//NO!!! sequence number will be managed internally by attention_head_io
	kv_offset *= max_tokens;			//each layer/key area in the cache is a sub-block, made up of "sequence_maxtokens" positions
	//NOW: we remove the following line from the offset: that's because the attention will process all the keys and values in the cache, starting from pos 0
	//kv_offset += pos;				//NO!!! position will be managed internally by attention_head_io
	kv_offset *= dim_qkv;			//each layer/key/pos in the cache is a key vector, made up of "dim_qkv" components
	//since the cache is actually "float *", we need to multiply by the size of float32
	kv_offset *= sizeof(float);


	byte * ks = &model->fm.keys[layer][kv_offset];
	byte * vs = &model->fm.values[layer][kv_offset];

	byte * dks = &model->grads.dkeys[layer][kv_offset];
	byte * dvs = &model->grads.dvalues[layer][kv_offset];


	size_t scores_offset = head * (sizeof(float) * T * T * B);




	//let's backprop attention!

mylog(LOG_VERBOSE_DEBUG,"let's backprop attention!");

	attention_head_io_backward(
	  dq, dks, dvs,
	  q, ks, vs,
	  0, 0, T, &model->grads.dheads_output[layer][0+(head*dim_qkv*sizeof(float))],
	  &model->fm.attention_scores[layer][scores_offset], &model->fm.raw_attention_scores[layer][scores_offset],
	  &model->grads.dattention_scores[layer][scores_offset], &model->grads.draw_attention_scores[layer][scores_offset],
	  model, attention_type, B, T
	);	



/*
if(head==10){
wdebug(q  + (1*T*dim_qkv*sizeof(float)) + (2*dim_qkv*sizeof(float)), WTYPE_FLOAT32, 5, "q", -1, 0);
wdebug(dq + (1*T*dim_qkv*sizeof(float)) + (2*dim_qkv*sizeof(float)), WTYPE_FLOAT32, 5, "dq", -1, 1);
}
*/




	//OFFSET CALCULATIONS (for key cache and value cache, and for weight matrixes)

	//1) offset for the current query multiplication matrix
	//   the offset would be the same for the output projection matrix, if it would be stored separately for each head, but this is never true at my current knowledge
	size_t qom_offset = 0;

	qom_offset += head;				//currently, we are processing query #head (over n_queries in total)
	qom_offset *= (dim_stream*dim_qkv);		//each matrix is (dim_stream,dim_qkv)
	//since the tensors are "byte *", we need to multiply by the size of the model datatype
	qom_offset *= wsize;



	//2) offset for the key and value multiplication matrixes
	size_t kvm_offset = 0;

	kvm_offset += kv;				//currently, we are processing key #kv (over n_keys in total)
	kvm_offset *= (dim_stream*dim_qkv);		//each matrix is (dim_stream,dim_qkv)
	//since the tensors are "byte *", we need to multiply by the size of the model datatype
	kvm_offset *= wsize;






	//let's backprop the COMPUTATION of the QUERY VECTOR


	if(last_qhead != head){


		//let's apply RoPE positional embeddings, if this is the case
		if(model->config.pose_cfg == POSE_ROPE){

			if(last_qhead != head){
				
mylog(LOG_VERBOSE_DEBUG,"let's backprop query ROPE");
				positional_embeddings_backward(
					dq, NULL,				
					dq, 
					model, dim_qkv, 0,
					B, T
				);
				
			}
		}


mylog(LOG_VERBOSE_DEBUG,"let's backprop query calculation (matmul+bias)");
		if(model->config.bias_cfg[3] == BIAS_ON){
		
			//#pragma omp parallel for collapse(2)	//NO! There would be a race over model->grads.dqb
			for(size_t b = 0; b < B; b++){
			    for(size_t ppos = 0; ppos < T; ppos++){

				size_t bto = ((b*T)+ppos)*dim_qkv*sizeof(float);	//offset

				sum_vectors_backward(
					&dq[bto], &model->grads.dqb[layer][head*dim_qkv*sizeof(float)],
					&dq[bto],
					dim_qkv
				);
			    }
			}
		}



		if((checkpoint_type==CP_SAFETENSORS)  && (model->config.pose_cfg!=POSE_LEARNED)  ){

		  matmulf_nt_interleaved_backward(
			&model->grads.dqm[layer][qom_offset], &model->grads.dnorm_pre_stream[layer][0],
			&model->w.qm[layer][qom_offset], &model->fm.norm_pre_stream[layer][0], dim_stream,dim_qkv, dim_stream,(B*T), dq,
			B, T
		  );

		}
		else{
		  matmulf_nt_backward(
			&model->grads.dqm[layer][qom_offset], &model->grads.dnorm_pre_stream[layer][0],
			&model->w.qm[layer][qom_offset], &model->fm.norm_pre_stream[layer][0], dim_stream,dim_qkv, dim_stream,(B*T), dq,
			B, T
		  );

		}



//if(head==1) wdebug(&q[dim_qkv*sizeof(float)*4], WTYPE_FLOAT32, dim_qkv, "q @pos 4 layer 0 head 1 before ROPE", -1, 1);

	}



	//3) offset of the current key vector (current layer, current head's key, current position of the sequence) in the model->fm.keys;
	//   the offset is the same for the value vector
	
	//IMPORTANT NOTE about key cache and value cache:
	//at each layer, for each head, we put all the keys (from position <zero> to <maxseq-1>) side by side, to allow matrix multiplications in the attention computation
	



	//let's backprop the COMPUTATION of the KEY VECTOR

	if(last_kvhead != kv){

	    //given the structure of the kv memory, I need to process one sequence at a time:  1,T   instead of B,T	(because of ROPE)
	    for(size_t b = 0; b < B; b++){

		kv_offset  = 0;
		kv_offset += kv;				//currently, we are processing key #kv (over n_keys in total)
		kv_offset *= B;				//each key area in the cache is a sub-block, made up of "B" sequences
		kv_offset += b;				//currently, we are processing sequence #b (over B sequences in total)
		kv_offset *= max_tokens;			//each sequence/key area in the cache is a sub-sub-block, made up of "sequence_maxtokens" positions
		//kv_offset += pos;				//currently, we are processing position #pos (over sequence_maxtokens in total)
		kv_offset *= dim_qkv;			//each pos/sequence/key in the cache is a key vector, made up of "dim_qkv" components
		//since the cache is actually "float *", we need to multiply by the size of float32
		kv_offset *= sizeof(float);

		byte * k  = &model->fm.keys[layer][kv_offset];
		byte * dk = &model->grads.dkeys[layer][kv_offset];



		//let's apply RoPE positional embeddings, if this is the case
		if(model->config.pose_cfg == POSE_ROPE){

			if(last_kvhead != kv){
			

mylog(LOG_DEBUG,"let's backprop key ROPE");

				//given the structure of the kv memory, I need to apply pose to one sequence at a time:  1,T   instead of B,T	
				positional_embeddings_backward(
					dk, NULL,				
					dk, 
					model, dim_qkv, 0,
					1, T
				);
				
			}
		}


mylog(LOG_DEBUG,"let's backprop key calculation (matmul+bias)");

		if(model->config.bias_cfg[3] == BIAS_ON){
	
			size_t ppos;	
			//#pragma omp parallel for private(ppos)	//NO! There wouldbe a race over model->grads.dkb
			for(ppos = 0; ppos < T; ppos++){

				size_t bto = ppos*dim_qkv*sizeof(float);	//offset

				sum_vectors_backward(
					&dk[bto], &model->grads.dkb[layer][kv*dim_qkv*sizeof(float)],
					&dk[bto],
					dim_qkv
				);
			}
		}



		if((checkpoint_type==CP_SAFETENSORS)  && (model->config.pose_cfg!=POSE_LEARNED)  ){

		  matmulf_nt_interleaved_backward(
			&model->grads.dkm[layer][kvm_offset], &model->grads.dnorm_pre_stream[layer][(b*T)*dim_stream*sizeof(float)],
			&model->w.km[layer][kvm_offset], &model->fm.norm_pre_stream[layer][(b*T)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), dk,
			1, T
		  );

		}
		else{

		  matmulf_nt_backward(
			&model->grads.dkm[layer][kvm_offset], &model->grads.dnorm_pre_stream[layer][(b*T)*dim_stream*sizeof(float)],
			&model->w.km[layer][kvm_offset], &model->fm.norm_pre_stream[layer][(b*T)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), dk,
			1, T
		  );

		}


	    }


	}



	//let's backprop the COMPUTATION of the VALUE VECTOR


	if(last_kvhead != kv){

	    for(size_t b = 0; b < B; b++){

		kv_offset  = 0;
		kv_offset += kv;				//currently, we are processing key #kv (over n_keys in total)
		kv_offset *= B;				//each key area in the cache is a sub-block, made up of "B" sequences
		kv_offset += b;				//currently, we are processing sequence #b (over B sequences in total)
		kv_offset *= max_tokens;			//each sequence/key area in the cache is a sub-sub-block, made up of "sequence_maxtokens" positions
		//kv_offset += pos;				//currently, we are processing position #pos (over sequence_maxtokens in total)
		kv_offset *= dim_qkv;			//each pos/sequence/key in the cache is a key vector, made up of "dim_qkv" components
		//since the cache is actually "float *", we need to multiply by the size of float32
		kv_offset *= sizeof(float);

		byte * v  = &model->fm.values[layer][kv_offset];
		byte * dv = &model->grads.dvalues[layer][kv_offset];


mylog(LOG_DEBUG,"let's backprop value calculation (matmul+bias)");

		if(model->config.bias_cfg[3] == BIAS_ON){
	
			size_t ppos;	
			//#pragma omp parallel for private(ppos)	//NO! there would be a raace condition over model->grads.dvb
			for(ppos = 0; ppos < T; ppos++){

				size_t bto = ppos*dim_qkv*sizeof(float);	//offset

				sum_vectors_backward(
					&dv[bto], &model->grads.dvb[layer][kv*dim_qkv*wsize],
					&dv[bto],
					dim_qkv
				);
			}
		}



		matmulf_nt_backward(
			&model->grads.dvm[layer][kvm_offset], &model->grads.dnorm_pre_stream[layer][(b*T)*dim_stream*sizeof(float)],
			&model->w.vm[layer][kvm_offset], &model->fm.norm_pre_stream[layer][(b*T)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), dv,
			1, T
		);



	    }


	}





				last_qhead = head;
				last_kvhead = kv;
	

			}	//for( head < n_queries )	


/*
wdebug(&model->grads.dqb[layer][((3*dim_qkv)+30)*wsize], WTYPE_FLOAT32, 25, "model->grads.dqb[layer][0*dim_qkv*wsize]", -1, 0);
wdebug(&model->grads.dkb[layer][((3*dim_qkv)+30)*wsize], WTYPE_FLOAT32, 25, "model->grads.dkb[layer][0*dim_qkv*wsize]", -1, 0);
wdebug(&model->grads.dvb[layer][((3*dim_qkv)+30)*wsize], WTYPE_FLOAT32, 25, "model->grads.dvb[layer][0*dim_qkv*wsize]", -1, 1);

*/
#ifdef TRIP_DEBUG
mylog(LOG_VERBOSE_DEBUG,"NORM  model->grads.dnorm_pre_stream[layer]  = %.7f", tensorgrad_norm_squared(NULL, model->grads.dnorm_pre_stream[layer], dim_stream*B*T));
#endif

/*
wdebug(model->fm.norm_pre_stream[layer]+750*4, WTYPE_FLOAT32, 15, "model->fm.norm_pre_stream[layer]", -1, 0);
wdebug(model->grads.dnorm_pre_stream[layer]+750*4, WTYPE_FLOAT32, 15, "model->grads.dnorm_pre_stream[layer]", -1, 1);
*/

mylog(LOG_DEBUG,"let's backprop into the PRE ATTENTION normalization");

		//let's backprop into the PRE ATTENTION normalization (right at the beginning of each layer)
		switch(model->config.norm_cfg[0]){
	
		    case NORM_NONE:
			//just backpropagate from output to input
			sum_vectors_backward(
				model->grads.dresidualstream_layerstart[layer], model->grads.dnorm_pre_stream[layer],
				model->grads.dnorm_pre_stream[layer],
				(dim_stream * B * T)
			);	
			break;
						
		    case NORM_LAYERNORM: 	//LAYERNORM
			layernorm_backward(
				model->grads.dresidualstream_layerstart[layer],  model->grads.dnorm_pre_w[layer],  model->grads.dnorm_pre_b[layer],
				model->grads.dnorm_pre_stream[layer],  model->fm.residualstream_layerstart[layer],  model->w.norm_pre_w[layer],  model->fm.norm_pre_mean[layer],  model->fm.norm_pre_rstd[layer],
				B, T, dim_stream
			);
			break;
	
		    case NORM_RMSNORM:	//RMSNORM
			rmsnorm_backward(
				model->grads.dresidualstream_layerstart[layer],  model->grads.dnorm_pre_w[layer],
				model->grads.dnorm_pre_stream[layer],  model->fm.residualstream_layerstart[layer],  model->w.norm_pre_w[layer],  model->fm.norm_pre_rrms[layer],
				model, B, T, dim_stream
			);
	
	  		break;
	
		    default: 
			mylog(LOG_ERROR, "Backward function for PRE-ATTENTION normalization %02X not implemented. Exiting...", model->config.norm_cfg[0]);
			exit(-1);
			break;
		}


		if(runtime_actions & (1<<SIGUSR2))	return -1;


	}	//end of loop over the layers


#ifdef TRIP_DEBUG
wdebug(model->grads.dresidualstream_layerstart[0] + (((1*T)+2)*dim_stream*sizeof(float)), WTYPE_FLOAT32, 5, "model->grads.dresidualstream_layerstart[0]", -1, 0);
#endif


	//THEN, let's backprop through sinusoidal position encoding (Vaswani) / learned, if required
	if(	(model->config.pose_cfg == POSE_ORIGINAL)
		||
		(model->config.pose_cfg == POSE_LEARNED)
	){
	

mylog(LOG_DEBUG,"let's backprop into POSE (original/learned)");


		positional_embeddings_backward(
			model->grads.dresidualstream_layerstart[0], ((model->config.pose_cfg == POSE_LEARNED) ? model->grads.dlearned_pose_w : NULL),				
			model->grads.dresidualstream_layerstart[0], 
			model, dim_stream, 0,
			B, T
		);
	}


#ifdef TRIP_DEBUG
wdebug(model->grads.dlearned_pose_w + (2*dim_stream*sizeof(float)), WTYPE_FLOAT32, 5, "model->grads.dlearned_pose_w", -1, 1);
#endif

	//if it's a Google GEMMA architecture, let's backprop through the embeddings scale factor
	if( (model->config.architectures == ARCH_GEMMA_CAUSAL)
	    ||
	    ((model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL) && (model->config.submodel_type == MODELTYPE_DECODER))
	){

		float scale_factor = sqrtf((float)dim_stream);			

		multiply_vector(model->grads.dresidualstream_layerstart[0], (dim_stream * B * T), scale_factor, model->grads.dresidualstream_layerstart[0]);			

	}



	if(runtime_actions & (1<<SIGUSR2))	return -1;

/*
mylog(LOG_DEBUG,"let's  NOT  backprop into token embeddings");
sleep(2);
return -1;
*/

#ifdef TRIP_DEBUG
mylog(LOG_DEBUG,"let's backprop into token embeddings");
#endif

	token_embeddings_backward(
		model->grads.dembeddings,
		model->grads.dresidualstream_layerstart[0], intok,
		model, B, T
	);	


#ifdef TRIP_DEBUG
wdebug(model->grads.dembeddings + (  333  *dim_stream*sizeof(float)), WTYPE_FLOAT32, 5, "model->grads.dembeddings", -1, 1);
#endif



	mylog(LOG_DEBUG,"BACKWARD COMPLETE!");

}



int adamw_set_config(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay){

	adamw_cfg.learning_rate = learning_rate;
	adamw_cfg.beta1 = beta1;
	adamw_cfg.beta2 = beta2;
	adamw_cfg.epsilon = epsilon;
	adamw_cfg.weight_decay = weight_decay;
}


int adamw_init(){

	adamw_cfg.root = NULL;
}


adamw_mv * adamw_create_mv(byte * params, size_t size){

	adamw_mv * node = NULL;

	if(adamw_cfg.root == NULL){	//if there's no node yet, not even the root node, let's create the root node
		
		adamw_cfg.root = (adamw_mv *)myalloc(sizeof(adamw_mv));
		node = adamw_cfg.root;
	}
	else{	//let's go through the list, and create a new node at the end of it

		node = adamw_cfg.root;

		while(node->next != NULL){
			node = node->next;
		}

		node->next = (adamw_mv *)myalloc(sizeof(adamw_mv));
		node = node->next;
	}

	node->params = params;
	node->m = (float *)myalloc(size * sizeof(float));
	node->v = (float *)myalloc(size * sizeof(float));
	node->next = NULL;

	return node;
}


int adamw_free(){
	
	adamw_mv * node;
	adamw_mv * next;

	node = adamw_cfg.root;

	while(node != NULL){

		next = node->next;

		free(node->m);
		free(node->v);
		free(node);

		node = next;
	}	
}

int adamw_set_step(size_t step){

	adamw_cfg.step = step;
}



int adamw(byte * _params, byte * _grads, size_t size){


	if(_params == NULL)	return -1;

	adamw_mv * node;
	adamw_mv ** next;

	next = &(adamw_cfg.root);
	node = adamw_cfg.root;

	while(node != NULL){

		if(node->params == _params)	break;	//node found

		next = &(node->next);
		node = node->next;
	}

	//if we couldn't find a node related to the requested params, let's create one!
	if(node == NULL){
		*next = adamw_create_mv(_params, size);
		node = *next;
	}



	float * grads  = (float *)_grads;

	float learning_rate = adamw_cfg.learning_rate;
	float beta1 = adamw_cfg.beta1;
	float beta2 = adamw_cfg.beta2;
	float epsilon = adamw_cfg.epsilon;
	float weight_decay = adamw_cfg.weight_decay;
	size_t step = adamw_cfg.step;


	size_t i;
	#pragma omp parallel for private(i)
	for(i = 0; i < size; i++){

		float param;

		if(wtype == WTYPE_FLOAT32){
			param = (float)(((float *)_params)[i]);
		}
		else
		if(wtype == WTYPE_BF16){
			param = (float)(((__bf16 *)_params)[i]);
		}
		else
		if(wtype == WTYPE_FLOAT16){
			param = (float)(((_Float16 *)_params)[i]);
		}



		float grad  = grads[i];	

/*
	//OLD: per-gradient gradient clipping; now replaced by: global L2-norm gradient clipping

		//float grad_th = 0.15;
		float grad_th = 0.3;
		//float grad_th = 1.5;
		if((grad > grad_th)  ||  (grad < (-1.0*grad_th))){
			
			if(grad > grad_th)		grad = grad_th;
			else
			if(grad < (-1.0*grad_th))	grad = -1.0 * grad_th;	
		}
*/


		//let's update momentum and rmsprop for the current parameter
		float momentum = (beta1 * node->m[i])  +  ((1.0 - beta1) * grad);
		float rmsprop  = (beta2 * node->v[i])  +  ((1.0 - beta2) * grad * grad);

		node->m[i] = momentum;
		node->v[i] = rmsprop;

		//before we apply them to update the current parameter, let's do on-the-fly a bias-correction for both momentum and rmsprop; 
		//this is useful to avoid that, during first steps, they become too much zero-biased
		momentum = momentum / (1.0 - powf(beta1, step+1));
		rmsprop  = rmsprop  / (1.0 - powf(beta2, step+1));


		//now, let's update the current parameter
		param   -=   learning_rate   *   ((momentum / (sqrtf(rmsprop) + epsilon))  +  (weight_decay * param));
		////param   -=   learning_rate   *   ((momentum / (sqrtf(rmsprop + epsilon)))  +  (weight_decay * param));



		if(wtype == WTYPE_FLOAT32){
			((float *)_params)[i] = param;
		}
		else
		if(wtype == WTYPE_BF16){
			((__bf16 *)_params)[i] = param;
		}
		else
		if(wtype == WTYPE_FLOAT16){
			((_Float16 *)_params)[i] = param;
		}
	

	}
}



float cosine_annealing_lr(ssize_t step, ssize_t warmup_steps, ssize_t total_steps, float max_lr, float min_lr){

    //warmup phase: linear increase from 0 to max_lr
    if((warmup_steps > 0) && (step <= warmup_steps)){
        return (max_lr * ((float)step / (float)warmup_steps));
    }
    
    //cosine decay phase
    ssize_t decay_steps = total_steps - warmup_steps;
    ssize_t current_decay_step = step - warmup_steps;
    
    //if we exceed total steps: go flat on min_lr
    if(current_decay_step >= decay_steps){
        return min_lr;
    }
    
    //cosine annealing formula
    float progress = ((float)current_decay_step) / ((float)decay_steps);
    float cosine_factor = 0.5 * (1.0 + cos(M_PI * progress));
    
    return (min_lr  +  ((max_lr - min_lr) * cosine_factor));
}




float tensorgrad_norm_squared(byte * _params, byte * _grads, size_t size){

	//if(_params == NULL)	return -1;

	float * grads  = (float *)_grads;

	float tensorgrads_norm_squared = 0.0;

	size_t i;
	for(i = 0; i < size; i++){

		float grad  = grads[i];
	
		tensorgrads_norm_squared += (grad*grad);
		
		
	}

	char buf[16];
	sprintf(buf,"%.1f",tensorgrads_norm_squared);
	if((memcmp(buf,"inf",3)==0) || (memcmp(buf,"-inf",4)==0) || (memcmp(buf,"nan",3)==0)){

		mylog(LOG_VERBOSE_DEBUG, "WARNING: tensorgrads_norm_squared is %f!", tensorgrads_norm_squared);

		float max_grad = -1.0;
		size_t i, max_i = -1;
		for(i = 0; i < size; i++){
			
			float grad  = grads[i];
	                grad = fabs(grad);
			if(grad>max_grad){
				max_grad = grad;
				max_i = i;
			}
        	}

		mylog(LOG_VERBOSE_DEBUG, "         max grad is grad[%d] = %f!", max_i, grads[max_i]);

		print_stacktrace();
		exit(-1);			
	}


	return tensorgrads_norm_squared;
}


bool gradients_multiply(float k, byte * _params, byte * _grads, size_t size){

	//if(_params == NULL)	return -1;

	float * grads  = (float *)_grads;


	size_t i;
	#pragma omp parallel for private(i)
	for(i = 0; i < size; i++){

		grads[i] *= k;	
	}

	return true;
}



bool gradients_check(Model * model, size_t T){


	if(runtime_actions & (1<<SIGUSR2))	return false;	//can exit only here; don't do during model update!


	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t dim_stream = (size_t)(model->config.dim_stream);
	//size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
	size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
	size_t vocab_size = (size_t)(model->config.vocab_size);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_queries  = (size_t)(model->config.n_queries);
	size_t n_keys     = (size_t)(model->config.n_keys);

	size_t max_tokens = T;

	int ffn_gating;
	if((model->config.ffn_nl_type[1] == GATE_ON)  ||  (model->config.ffn_nl_type[0] == NL_SILU_LLAMA))	ffn_gating = 1;
	else													ffn_gating = 0;


	mylog(LOG_VERBOSE_DEBUG, "Checking gradients...");

	float global_norm = 0.0;


/*
	//extra tensors for vision model (encoder)
	byte * vision_embeddings_w;	// flattened_patch_size * dim_stream	
	byte * vision_embeddings_b;	// dim_stream
	byte * learned_pose_w;		// n_patches * dim_stream

	byte * multimodal_projector_w;	// dim_stream * target_dim_stream
	byte * multimodal_projector_b;	// 1          * target_dim_stream
 */


	//extra tensors for language model (decoder)

	//the order of tensors is backward here for debug purposes (intercept inf/Nan as soon as they appear in the gradients flow)
 
	global_norm += tensorgrad_norm_squared(model->w.logits_classifier, model->grads.dlogits_classifier, dim_stream * vocab_size);	// dim_stream * vocab_size

	mylog(LOG_VERBOSE_DEBUG, "after dlogits_classifier");
#ifdef TRIP_DEBUG
	mylog(LOG_VERBOSE_DEBUG, "partial global gradients L2-norm = %.6f",global_norm);
#endif



	//2) single-layer	tensors
	global_norm += tensorgrad_norm_squared(model->w.norm_final_w, model->grads.dnorm_final_w, dim_stream);	// 1 * dim_stream 
	global_norm += tensorgrad_norm_squared(model->w.norm_final_b, model->grads.dnorm_final_b, dim_stream);	// 1 * dim_stream

	mylog(LOG_VERBOSE_DEBUG, "after dnorm_final");
#ifdef TRIP_DEBUG
	mylog(LOG_VERBOSE_DEBUG, "partial global gradients L2-norm = %.6f",global_norm);
#endif



	//tensors for all model types (decoder or encoder)
	//1) per-layer tensors

	for(ssize_t layer = (n_layers-1); layer >= 0; layer--){

		mylog(LOG_VERBOSE_DEBUG, "layer #%d", layer);
#ifdef TRIP_DEBUG
		mylog(LOG_VERBOSE_DEBUG, "partial global gradients L2-norm = %.6f",global_norm);
#endif

		global_norm += tensorgrad_norm_squared(model->w.norm_post_w[layer], model->grads.dnorm_post_w[layer], dim_stream);	// n_layers * dim_stream 
		global_norm += tensorgrad_norm_squared(model->w.norm_post_b[layer], model->grads.dnorm_post_b[layer], dim_stream);	// n_layers * dim_stream
	
		global_norm += tensorgrad_norm_squared(model->w.post_ffn_w[layer], model->grads.dpost_ffn_w[layer], dim_stream * hidden_dim);	// n_layers * ffn_hidden_dim * dim_stream
		global_norm += tensorgrad_norm_squared(model->w.post_ffn_b[layer], model->grads.dpost_ffn_b[layer],          1 * dim_stream);		// n_layers * 1              * dim_stream   



		global_norm += tensorgrad_norm_squared(model->w.pre_ffn_w[layer], model->grads.dpre_ffn_w[layer], dim_stream * hidden_dim);	// n_layers * dim_stream * ffn_hidden_dim 
		global_norm += tensorgrad_norm_squared(model->w.pre_ffn_b[layer], model->grads.dpre_ffn_b[layer],          1 * hidden_dim);	// n_layers * 1          * ffn_hidden_dim

		if(ffn_gating == 1){
			global_norm += tensorgrad_norm_squared(model->w.pre_ffn_w2[layer], model->grads.dpre_ffn_w2[layer], dim_stream * hidden_dim);	// n_layers * dim_stream * ffn_hidden_dim 
		}


		global_norm += tensorgrad_norm_squared(model->w.om[layer], model->grads.dom[layer], dim_stream * dim_qkv * n_queries);		// dim_stream * dim_qkv * n_queries * n_layers

		global_norm += tensorgrad_norm_squared(model->w.vm[layer], model->grads.dvm[layer], dim_stream * dim_qkv * n_keys);		// dim_stream * dim_qkv * n_keys * n_layers 

		global_norm += tensorgrad_norm_squared(model->w.qm[layer], model->grads.dqm[layer], dim_stream * dim_qkv * n_queries);		// dim_stream * dim_qkv * n_queries * n_layers 
		global_norm += tensorgrad_norm_squared(model->w.km[layer], model->grads.dkm[layer], dim_stream * dim_qkv * n_keys);		// dim_stream * dim_qkv * n_keys * n_layers   


		if(model->config.bias_cfg[3] == BIAS_ON){

			global_norm += tensorgrad_norm_squared(model->w.qb[layer], model->grads.dqb[layer],          1 * dim_qkv * n_queries);		// 1          * dim_qkv * n_queries * n_layers 
			global_norm += tensorgrad_norm_squared(model->w.kb[layer], model->grads.dkb[layer],          1 * dim_qkv * n_keys);		// 1          * dim_qkv * n_keys * n_layers  
			global_norm += tensorgrad_norm_squared(model->w.vb[layer], model->grads.dvb[layer],          1 * dim_qkv * n_keys);		// 1          * dim_qkv * n_keys * n_layers
			global_norm += tensorgrad_norm_squared(model->w.ob[layer], model->grads.dob[layer],          1 * dim_qkv * n_queries);		// 1          * dim_qkv * n_queries * n_layers
		}

		global_norm += tensorgrad_norm_squared(model->w.norm_pre_w[layer], model->grads.dnorm_pre_w[layer], dim_stream);
		global_norm += tensorgrad_norm_squared(model->w.norm_pre_b[layer], model->grads.dnorm_pre_b[layer], dim_stream);	// n_layers * dim_stream
	
	}

	if(model->config.pose_cfg == POSE_LEARNED){

		global_norm += tensorgrad_norm_squared(model->w.learned_pose_w, model->grads.dlearned_pose_w, max_tokens * dim_stream);		// max_tokens * dim_stream
	}


	if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){

		global_norm += tensorgrad_norm_squared(model->w.embeddings, model->grads.dembeddings, vocab_size * dim_stream);		// vocab_size * dim_stream 
	}






	global_norm = sqrtf(global_norm);

	mylog(LOG_VERBOSE_DEBUG,"Global gradients L2-norm = %.6f",global_norm);

	float global_norm_th = 1.0;

	mylog(LOG_VERBOSE_DEBUG,"Global gradients L2-norm threshold = %.6f",global_norm_th);

	if(global_norm > global_norm_th){

		mylog(LOG_VERBOSE_DEBUG,"Applying gradient clipping...");

		float k = (global_norm_th / global_norm);

	
	/*
		//extra tensors for vision model (encoder)
		byte * vision_embeddings_w;	// flattened_patch_size * dim_stream	
		byte * vision_embeddings_b;	// dim_stream
		byte * learned_pose_w;		// n_patches * dim_stream
	
		byte * multimodal_projector_w;	// dim_stream * target_dim_stream
		byte * multimodal_projector_b;	// 1          * target_dim_stream
	 */
	
	
		//extra tensors for language model (decoder)
	
	
		if(model->config.pose_cfg == POSE_LEARNED){
	
			gradients_multiply(k, model->w.learned_pose_w, model->grads.dlearned_pose_w, max_tokens * dim_stream);		// max_tokens * dim_stream
		}
	
	 
		if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){
	
			gradients_multiply(k, model->w.embeddings, model->grads.dembeddings, vocab_size * dim_stream);		// vocab_size * dim_stream 
	
			gradients_multiply(k, model->w.logits_classifier, model->grads.dlogits_classifier, dim_stream * vocab_size);	// dim_stream * vocab_size
		}
		else{
	
			//IMPORTANT:	when embeddings are tied (i.e: when the logits classifier is actually the same structure of the word embeddings),
			//		the update must be applied only ONCE!
	
			gradients_multiply(k, model->w.embeddings, model->grads.dembeddings, vocab_size * dim_stream);		// vocab_size * dim_stream 
		}
	
	
		//tensors for all model types (decoder or encoder)
		//1) per-layer tensors
	
		for(ssize_t layer = 0; layer < n_layers; layer++){
	
	mylog(LOG_DEBUG, "layer #%d", layer);
	
			gradients_multiply(k, model->w.norm_pre_w[layer], model->grads.dnorm_pre_w[layer], dim_stream);
			gradients_multiply(k, model->w.norm_pre_b[layer], model->grads.dnorm_pre_b[layer], dim_stream);	// n_layers * dim_stream
	
			gradients_multiply(k, model->w.qm[layer], model->grads.dqm[layer], dim_stream * dim_qkv * n_queries);		// dim_stream * dim_qkv * n_queries * n_layers 
			gradients_multiply(k, model->w.km[layer], model->grads.dkm[layer], dim_stream * dim_qkv * n_keys);		// dim_stream * dim_qkv * n_keys * n_layers   
			gradients_multiply(k, model->w.vm[layer], model->grads.dvm[layer], dim_stream * dim_qkv * n_keys);		// dim_stream * dim_qkv * n_keys * n_layers 
			gradients_multiply(k, model->w.om[layer], model->grads.dom[layer], dim_stream * dim_qkv * n_queries);		// dim_stream * dim_qkv * n_queries * n_layers
	
			if(model->config.bias_cfg[3] == BIAS_ON){
	
				gradients_multiply(k, model->w.qb[layer], model->grads.dqb[layer],          1 * dim_qkv * n_queries);		// 1          * dim_qkv * n_queries * n_layers 
				gradients_multiply(k, model->w.kb[layer], model->grads.dkb[layer],          1 * dim_qkv * n_keys);		// 1          * dim_qkv * n_keys * n_layers  
				gradients_multiply(k, model->w.vb[layer], model->grads.dvb[layer],          1 * dim_qkv * n_keys);		// 1          * dim_qkv * n_keys * n_layers
				gradients_multiply(k, model->w.ob[layer], model->grads.dob[layer],          1 * dim_qkv * n_queries);		// 1          * dim_qkv * n_queries * n_layers
			}
	
			gradients_multiply(k, model->w.norm_post_w[layer], model->grads.dnorm_post_w[layer], dim_stream);	// n_layers * dim_stream 
			gradients_multiply(k, model->w.norm_post_b[layer], model->grads.dnorm_post_b[layer], dim_stream);	// n_layers * dim_stream
		
			gradients_multiply(k, model->w.pre_ffn_w[layer], model->grads.dpre_ffn_w[layer], dim_stream * hidden_dim);	// n_layers * dim_stream * ffn_hidden_dim 
			gradients_multiply(k, model->w.pre_ffn_b[layer], model->grads.dpre_ffn_b[layer],          1 * hidden_dim);	// n_layers * 1          * ffn_hidden_dim
	
			if(ffn_gating == 1){
				gradients_multiply(k, model->w.pre_ffn_w2[layer], model->grads.dpre_ffn_w2[layer], dim_stream * hidden_dim);	// n_layers * dim_stream * ffn_hidden_dim 
			}
	
			gradients_multiply(k, model->w.post_ffn_w[layer], model->grads.dpost_ffn_w[layer], dim_stream * hidden_dim);	// n_layers * ffn_hidden_dim * dim_stream
			gradients_multiply(k, model->w.post_ffn_b[layer], model->grads.dpost_ffn_b[layer],          1 * dim_stream);		// n_layers * 1              * dim_stream   
		
		}
	
		//2) single-layer	tensors
		gradients_multiply(k, model->w.norm_final_w, model->grads.dnorm_final_w, dim_stream);	// 1 * dim_stream 
		gradients_multiply(k, model->w.norm_final_b, model->grads.dnorm_final_b, dim_stream);	// 1 * dim_stream
	

	}
		
	return true;
}



bool model_update(Model * model, size_t step, size_t T){


	if(runtime_actions & (1<<SIGUSR2))	return false;	//can exit only here; don't do during model update!


	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t dim_stream = (size_t)(model->config.dim_stream);
	//size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
	size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
	size_t vocab_size = (size_t)(model->config.vocab_size);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_queries  = (size_t)(model->config.n_queries);
	size_t n_keys     = (size_t)(model->config.n_keys);

	size_t max_tokens = T;

	int ffn_gating;
	if((model->config.ffn_nl_type[1] == GATE_ON)  ||  (model->config.ffn_nl_type[0] == NL_SILU_LLAMA))	ffn_gating = 1;
	else													ffn_gating = 0;


	mylog(LOG_VERBOSE_DEBUG, "Entering model_update(step=%d)!", step);

	adamw_set_step(step);

/*
	//extra tensors for vision model (encoder)
	byte * vision_embeddings_w;	// flattened_patch_size * dim_stream	
	byte * vision_embeddings_b;	// dim_stream
	byte * learned_pose_w;		// n_patches * dim_stream

	byte * multimodal_projector_w;	// dim_stream * target_dim_stream
	byte * multimodal_projector_b;	// 1          * target_dim_stream
 */


	//extra tensors for language model (decoder)


	if(model->config.pose_cfg == POSE_LEARNED){

		adamw(model->w.learned_pose_w, model->grads.dlearned_pose_w, max_tokens * dim_stream);		// max_tokens * dim_stream
	}

 
	if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){

		adamw(model->w.embeddings, model->grads.dembeddings, vocab_size * dim_stream);		// vocab_size * dim_stream 

		adamw(model->w.logits_classifier, model->grads.dlogits_classifier, dim_stream * vocab_size);	// dim_stream * vocab_size
	}
	else{

		//IMPORTANT:	when embeddings are tied (i.e: when the logits classifier is actually the same structure of the word embeddings),
		//		the update must be applied only ONCE!

		adamw(model->w.embeddings, model->grads.dembeddings, vocab_size * dim_stream);		// vocab_size * dim_stream 
	}


	//tensors for all model types (decoder or encoder)
	//1) per-layer tensors

	for(size_t layer = 0; layer < n_layers; layer++){

mylog(LOG_DEBUG, "layer #%d", layer);

		adamw(model->w.norm_pre_w[layer], model->grads.dnorm_pre_w[layer], dim_stream);
		adamw(model->w.norm_pre_b[layer], model->grads.dnorm_pre_b[layer], dim_stream);	// n_layers * dim_stream

		adamw(model->w.qm[layer], model->grads.dqm[layer], dim_stream * dim_qkv * n_queries);		// dim_stream * dim_qkv * n_queries * n_layers 
		adamw(model->w.km[layer], model->grads.dkm[layer], dim_stream * dim_qkv * n_keys);		// dim_stream * dim_qkv * n_keys * n_layers   
		adamw(model->w.vm[layer], model->grads.dvm[layer], dim_stream * dim_qkv * n_keys);		// dim_stream * dim_qkv * n_keys * n_layers 
		adamw(model->w.om[layer], model->grads.dom[layer], dim_stream * dim_qkv * n_queries);		// dim_stream * dim_qkv * n_queries * n_layers

		if(model->config.bias_cfg[3] == BIAS_ON){

			adamw(model->w.qb[layer], model->grads.dqb[layer],          1 * dim_qkv * n_queries);		// 1          * dim_qkv * n_queries * n_layers 
			adamw(model->w.kb[layer], model->grads.dkb[layer],          1 * dim_qkv * n_keys);		// 1          * dim_qkv * n_keys * n_layers  
			adamw(model->w.vb[layer], model->grads.dvb[layer],          1 * dim_qkv * n_keys);		// 1          * dim_qkv * n_keys * n_layers
			adamw(model->w.ob[layer], model->grads.dob[layer],          1 * dim_qkv * n_queries);		// 1          * dim_qkv * n_queries * n_layers
		}

		adamw(model->w.norm_post_w[layer], model->grads.dnorm_post_w[layer], dim_stream);	// n_layers * dim_stream 
		adamw(model->w.norm_post_b[layer], model->grads.dnorm_post_b[layer], dim_stream);	// n_layers * dim_stream
	
		adamw(model->w.pre_ffn_w[layer], model->grads.dpre_ffn_w[layer], dim_stream * hidden_dim);	// n_layers * dim_stream * ffn_hidden_dim 
		adamw(model->w.pre_ffn_b[layer], model->grads.dpre_ffn_b[layer],          1 * hidden_dim);	// n_layers * 1          * ffn_hidden_dim

		if(ffn_gating == 1){
			adamw(model->w.pre_ffn_w2[layer], model->grads.dpre_ffn_w2[layer], dim_stream * hidden_dim);	// n_layers * dim_stream * ffn_hidden_dim 
		}

		adamw(model->w.post_ffn_w[layer], model->grads.dpost_ffn_w[layer], dim_stream * hidden_dim);	// n_layers * ffn_hidden_dim * dim_stream
		adamw(model->w.post_ffn_b[layer], model->grads.dpost_ffn_b[layer],          1 * dim_stream);		// n_layers * 1              * dim_stream   
	
	}

	//2) single-layer	tensors
	adamw(model->w.norm_final_w, model->grads.dnorm_final_w, dim_stream);	// 1 * dim_stream 
	adamw(model->w.norm_final_b, model->grads.dnorm_final_b, dim_stream);	// 1 * dim_stream


		
	return true;
}

