#define TRIP_FORWARD_VERSION 2026041501
#include "trip.h"

int compare_logit_elements(const void * aa, const void * bb){
	LogitElement * a = (LogitElement *)aa;
	LogitElement * b = (LogitElement *)bb;

	if(a->prob > b->prob)	return -1;
	else
	if(b->prob > a->prob)	return +1;
	else			return  0;
}


//this function takes as input the distribution probability over all the vocabulary entries generated as output of the final layer
//and chooses the token, following the rules/parameters specified by the user:
// temperature
// AND
// top_p  OR  top_k

int sample_next_token(Model * model, byte * _logits, float * out_prob){

	size_t vocab_size = model->config.vocab_size;


	if(checkpoint_type == CP_GPT2_AK){

		float * logits = (float *)_logits;

		size_t nonpadded_vocab_size = 50257;
		for(size_t i = nonpadded_vocab_size; i < vocab_size; i++){
			logits[i] = 0.0;
		}

		vocab_size = nonpadded_vocab_size;
	}	


	if(temperature == 0.0){	//this is a way to say that we just want to sample the vocabulary entry with the highest calculated probability

		int max_prob_i = -1;


		float * logits = (float *)_logits;	
	   	float max_prob = -FLT_MAX;

		if(out_prob!=NULL){
			//softmax must be applied anyway, 
			//since the input logit vector is not yet normalized to a probability distribution (i.e: with sum over probabilities = 1.0)
			//and the caller is asking to return also the probability for the sampled token
			softmax(_logits, 1, vocab_size, _logits);
		}
	
		for(int i=0; i < vocab_size; i++){

			if(logits[i] > max_prob){
				max_prob = logits[i];	//NOTE: we call it "probability", but the logits may NOT be normalized to sum up to 1.0 (if we did not use softmax, see right above)
				max_prob_i = i;		//      but to our current purpose this is not relevant, if we were not required to return the probabilty of the sampled token
			}
		}
	
		if(out_prob!=NULL){
			*out_prob = logits[max_prob_i];
		}

		return max_prob_i;

	}
	else{	//if temperature is not 0.0
	
		float * logits = (float *)_logits;

		if(temperature != 1.0){

			for(int i=0; i < vocab_size; i++){
				logits[i] /= temperature;
			}
		}

		//softmax must be applied anyway, since the input logit vector is not yet normalized to a probability distribution (i.e: with sum over probabilities = 1.0)
		softmax(_logits, 1, vocab_size, _logits);
	

		if(top_k > 0){	//if top_k has been activated by the user (not default)

			if(top_k > vocab_size){
				mylog(LOG_INFO,"WARNING: capping specified top_k (%d) down to vocab_size (%d).", top_k, vocab_size);
				top_k = vocab_size;
			}

			//we need to store the logits elements in an array of objects which can keep track of their original vocabulary index
			int ilogits_size = vocab_size;
			LogitElement * IndexedLogits = (LogitElement *)myalloc(ilogits_size * sizeof(LogitElement));
			for(int i=0; i < ilogits_size; i++){
				float logits_i = (((float *)_logits)[i]);
				IndexedLogits[i].prob = logits_i; 
				IndexedLogits[i].index = i;
			}

			//now we sort them: the first element will be the one with the highest probability, and then elements with lower probabilities, decreasing...
			qsort(IndexedLogits, ilogits_size, sizeof(LogitElement), compare_logit_elements);


			//we choose a random value [0,1), but then we normalize it to the sum of the probabilities of the first "top_k" elements
			float cum_prob = 0.0;
			for(int i=0; i < top_k; i++){
				cum_prob += IndexedLogits[i].prob;
			}

			if(cum_prob <= 0.0){
				mylog(LOG_ERROR,"first top_k (%d) probabilities have bad sum-up (%.5f). Exiting...", top_k, cum_prob);
				exit(-1);
			}

			float rn;
			rn  = ((float)rand()) / ((float)RAND_MAX);	// [ 0.000 , 1.000 )
			rn *= cum_prob;


			//now we are ready to sample the logit entry
			cum_prob = 0.0;
			int ii;
			for(int i=0; i < top_k; i++){
				cum_prob += IndexedLogits[i].prob;
				//if the cumulative probability is higher than our random point, we are done
				if(cum_prob > rn){
					ii = i;	
					break;
				}
			}

			size_t max_prob_i = IndexedLogits[ii].index;

			//let's free up the dynamically allocated memory
			free(IndexedLogits);
	
			if(out_prob!=NULL){
				*out_prob = logits[max_prob_i];
			}

		

			//let's return the original index of the current logit element! (it's the index of the vocabulary entry)
			return max_prob_i;

		}
		else{	//if we use top_p (i.e.: nucleus sampling)

			if((top_p > 1.0)  ||  (top_p <= 0.0)){
				mylog(LOG_INFO,"WARNING: specified top_p (%.3f) is out of range. Defaulting to %.3f .", top_p, DEFAULT_TOP_P);
				top_p = DEFAULT_TOP_P;
			}

		
			//now, the following is just a heuristic (a good one!), to lower down the computation due to qsort: 
			//we can avoid to put into the array of logit elements to be sorted by qsort
			//all the elements with probability lower than "reference_p", because for sure they will be out 
			//of the set of highest probability elements, whose probabilities sum up to top_p
			float residual_p = (1.0 - top_p);
			float reference_p = residual_p / (vocab_size - 1);

			//we need to store the logits elements in an array of objects which can keep track of their original vocabulary index
			int ilogits_size = 0;
			LogitElement * IndexedLogits = (LogitElement *)myalloc(vocab_size * sizeof(LogitElement));

			for(int i=0; i < vocab_size; i++){
				float logits_i = (((float *)_logits)[i]);
				//if the probability of the current element of the orginal logits vector
				//is higher than the calculated probaility threshold, 
				//let's store this element into the indexed vector
				if(logits_i >= reference_p){
					IndexedLogits[ilogits_size].prob  = logits_i;
					IndexedLogits[ilogits_size].index = i;
					ilogits_size++;
				}
				//otherwise, just skip this element
			}

			//now we sort them: the first element will be the one with the highest probability, and then elements with lower probabilities, decreasing...
			qsort(IndexedLogits, ilogits_size, sizeof(LogitElement), compare_logit_elements);


			//we choose a random value [0,1), but then we normalize it to the specified "top_p" probability
			float rn;
			rn  = ((float)rand()) / ((float)RAND_MAX);	// [ 0.000 , 1.000 )
			rn *= top_p;	// [ 0.000 , top_p )


			//now we are ready to sample the logit entry
			float cum_prob = 0.0;
			int ii;
			for(int i=0; i < ilogits_size; i++){
				cum_prob += IndexedLogits[i].prob;
				//if the cumulative probability is higher than our random point, we are done
				if(cum_prob > rn){
					ii = i;
					break;
				}
			}


			size_t max_prob_i = IndexedLogits[ii].index;

			//let's free up the dynamically allocated memory
			free(IndexedLogits);
	
			if(out_prob!=NULL){
				*out_prob = logits[max_prob_i];
			}

		

			//let's return the original index of the current logit element! (it's the index of the vocabulary entry)
			return max_prob_i;


		}	

	}
}


void wdebug(byte * _in, int wtype_in, size_t dim, char * label, ssize_t index, int terminate){

	mylog(LOG_VERBOSE_DEBUG, label, index);

   if(wtype_in == WTYPE_FLOAT32){
	float * in = (float *)_in;
	for(size_t i=0; i<dim; i++){
		if((i!=0)&&((i%10)==0))    printf("\n");
		printf("%04zd: %9.6f   ",i,in[i]);
	}
   }
   else
   if(wtype_in == WTYPE_BF16){
	__bf16 * in = (__bf16 *)_in;
        for(size_t i=0; i<dim; i++){
		if((i!=0)&&((i%10)==0))    printf("\n");
                printf("%04zd: %9.6f   ",i,(float)in[i]);
        }
   }
   else
   if(wtype_in == WTYPE_FLOAT16){
	_Float16 * in = (_Float16 *)_in;
        for(size_t i=0; i<dim; i++){
		if((i!=0)&&((i%10)==0))    printf("\n");
                printf("%04zd: %9.6f   ",i,(float)in[i]);
        }
   }


	printf("\n\n");
	if(terminate>0)      exit(1);
}

// ============================================================
//  forward()
//
//  This is where it all happens: the sequence of token IDs
//  (the tokenized text) is processed: one by one, they are
//  translated to embeddings — vectors in the internal
//  "thinking space" of the transformer — and injected into
//  the residual stream.
//
//  The residual stream is the bus: data flows through it,
//  and each layer reads from it and writes back to it.
//
//  The flow, at each position in the sequence:
//
//  1. Look up the token embedding from the vocabulary table.
//     The embedding IS the initial residual stream.
//
//  2. Apply positional encoding (learned, or sinusoidal;
//     if RoPE, they will be applied later, directly on
//     queries and keys — see below) so the model knows
//     WHERE each token sits in the sequence.
//
//  3. Walk through all N layers. Each layer does:
//     a) PRE-ATTENTION NORM  — branch (make a copy of) the
//        stream, then normalize the branched copy
//        (RMSnorm or LayerNorm)
//     b) ATTENTION — project Q, K, V for each head,
//        apply RoPE to Q and K (if positional embeddings
//        are RoPE), compute attention scores
//        (Q·K^T / sqrt(d)), apply causal mask, softmax,
//        multiply by V, project output
//     c) RESIDUAL ADD — add the attention output back to
//        the stream
//     d) POST-ATTENTION NORM — branch the stream again,
//        and normalize (some architectures skip this)
//     e) FFN — up-project (a matrix multiply, expanding
//        the dimension of the working vector), apply
//        non-linearity (SiLU/GELU/ReLU), optionally gate,
//        down-project (going back to residual stream dim)
//     f) RESIDUAL ADD — add the FFN output back to the
//        stream
//
//  4. FINAL NORM — normalize the output of the last layer.
//     This is the only point in which the main residual
//     stream (the bus) gets normalized.
//
//  5. CLASSIFY — project the stream to vocabulary size
//     (logits). This is actually just a proxy to cosine
//     similarity, or if you prefer a pattern-matching
//     process: which vocabulary embedding vectors give the
//     largest dot product with the residual stream?
//     (This deserves some meditation about the inner
//     workings of a transformer. E.g: vocabulary embeddings
//     lying on the same exact direction can't coexist
//     because the larger one will always be chosen — or,
//     if they exist, the smaller ones will always serve as
//     fallback/second choices when not choosing
//     deterministically, i.e. when not "greedy sampling".)
//
//  6. SAMPLE — pick the next token (greedy, top-k, top-p).
//     In training mode, steps 5-6 run for ALL positions in
//     parallel (teacher forcing). In inference, when
//     generating new tokens, we loop one position at a
//     time, feeding each sampled token as input to the next
//     step. When processing the prompt instead, all tokens
//     can be processed in parallel, because the prediction
//     at position i will not overwrite the prompt token at
//     position i+1. Essentially, the prompt colonizes the
//     residual stream. So: why process the prompt at all?
//     Because the processing builds the KV cache, which is
//     the full trace of the transformer's internal
//     processing, and will be used by ALL future positions
//     — nothing goes lost, only the output residual stream
//     is discarded when the next position is still a prompt
//     token.
//
// ============================================================

int forward(Model * model, byte * embeddings, byte * out_streams, int ** intok, int * n_intoks, size_t pos, int attention_type, size_t B, size_t T){

	size_t b;	//index for batch
	size_t ppos;	//index for token

	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t dim_stream = (size_t)(model->config.dim_stream);
	size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
	size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
	size_t vocab_size = (size_t)(model->config.vocab_size);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_queries  = (size_t)(model->config.n_queries);
	size_t n_keys     = (size_t)(model->config.n_keys);


	if(action == ACTION_TRAIN){

		//T = T - 1;
		max_tokens = T;

		//when calling forward, we have set "T" to "max tokens - 1"  because, when training, the last token in the input sequence has not to be processed;
		//it serves only as the target against which we will calculate loss, 
		//given the predicted probability calculated for that next "target" token after the processing of the current token.
	}


	size_t answer_startpos = pos + T;

	//if(T > max_tokens)		T = max_tokens;
	if(T < 1)			T = 1;
	if((attention_type == ATTENTION_FULL)  &&  (parallel_forwarding == PARALLEL_FORWARDING_OFF)){
		mylog(LOG_INFO,"Model architecture (encoder) requires parallel forwarding ON. Forcing...");
		parallel_forwarding = PARALLEL_FORWARDING_ON;
		T = max_tokens;
	}
	if(parallel_forwarding == PARALLEL_FORWARDING_OFF)	T = 1;


	mylog(LOG_INFO,"Parallel forwarding = %s,   steps = %d",((parallel_forwarding==0)?"off":"ON"),T);


	if(action != ACTION_TRAIN){	//if we are performing training, we manage the allocation of the forward memory outside this function
		alloc_forward_memory(model, B, T, action);
	}


	//for sake of short notations:	
	byte ** residualstream_layerstart 	= model->fm.residualstream_layerstart;

	byte ** norm_pre_stream 		= model->fm.norm_pre_stream;
	byte ** queries 			= model->fm.queries;
	byte ** heads_output 			= model->fm.heads_output;

	byte ** attentionlayer_out_stream	= model->fm.attentionlayer_out_stream;

	byte ** residualstream_after_attention 	= model->fm.residualstream_after_attention;

	byte ** norm_post_stream 		= model->fm.norm_post_stream;
	byte ** ffn_in_stream 			= model->fm.ffn_in_stream;
	byte ** ffn_aux_stream 			= model->fm.ffn_aux_stream;
	byte ** ffn_out_stream 			= model->fm.ffn_out_stream;
	byte ** ffn_final_stream 		= model->fm.ffn_final_stream;

	byte ** residualstream_after_ffn 	= model->fm.residualstream_after_ffn;

	byte * norm_final_stream 		= model->fm.norm_final_stream;

	byte * logits 				= model->fm.logits;


	start_ts = get_milliseconds();
	last_ts  = start_ts;
	


	while(pos < max_tokens){


		
#define SPEEDTEST_NTOKS	10

	    if((pos>0) && ((pos%SPEEDTEST_NTOKS)==0)){
			long int curr_ts = get_milliseconds();

			long int total_time = curr_ts - start_ts;
			long int last_time  = curr_ts - last_ts;

			mylog(LOG_DEBUG,"\t\t\t\t\t\t\t tok/s = %.3f \t tok/s from start = %.3f (total: %d tokens in %.6f s)", 
					((((float)SPEEDTEST_NTOKS)*1000.0)/(float)last_time), 
					((((float)pos            )*1000.0)/(float)total_time),
					pos,
					(((float)total_time)/1000.0));

			last_ts = curr_ts;	
	    }

#ifdef TRIP_DEBUG
	    mylog(LOG_DEBUG,"currts=%ld",get_milliseconds());
#endif



	if( 	(action == ACTION_DECODE)
		||
		(action == ACTION_CHAT)
		||
		((action == ACTION_VISION) && (model->config.submodel_type == MODELTYPE_DECODER))
		||
		(action == ACTION_TRAIN)
	){

	  for(b = 0; b < B ; b++){
	    for(ppos = pos; ppos < (pos + T); ppos++){

		if(log_cfg==LOG_INFO){
			if(	((action == ACTION_CHAT) && (ppos >= answer_startpos))
				||
				(action == ACTION_DECODE)
				||
				((action == ACTION_VISION) && (ppos >= answer_startpos))
			){
				md_printf("%s",toki.vocab[intok[b][ppos]]);
				fflush(stdout);

			}
		}
		else{
			char report_prefix[256];

			if(B>1){	
				//if this is a batch, and we are processing one of the sequences in the batch:
				sprintf(report_prefix,"batch_seq %.5zd  ",b);
			}	
			else{
				report_prefix[0] = '\0';
			}

			mylog(LOG_DEBUG,"%spos %.5d:\ttoken = %.5d\t%s", report_prefix, ppos, intok[b][ppos], toki.vocab[intok[b][ppos]]);
		}


		if((action == ACTION_VISION) && (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)){

			//object detection
			if((intok[b][ppos] >= 256000) && (intok[b][ppos] <= 257023)){

				if((vision_detect_status<0) || (vision_detect_status>=4)){
					vision_detect_status = 0;
				}
				switch(vision_detect_status){
				case 0:	vision_detect_y_min = intok[b][ppos] - 256000;	break;
				case 1:	vision_detect_x_min = intok[b][ppos] - 256000;	break;
				case 2:	vision_detect_y_max = intok[b][ppos] - 256000;	break;
				case 3:	vision_detect_x_max = intok[b][ppos] - 256000;	break;
				}
				vision_detect_status++;
				if(vision_detect_status==4){
					free_forward_memory(model);
					return ppos;
				}
			}
			else{
				vision_detect_status = 0;
			}
		}
	

	    }
	  }
	}


	//if the forward() function has been called without specifying a starting vector of embeddings vectors (e.g: text decoding, chat mode)
	if((embeddings == NULL)  ||  (pos>=answer_startpos)){

	  #pragma omp parallel for collapse(2) 
	  for(size_t b = 0; b < B ; b++){
	    for(size_t ppos = 0; ppos < T; ppos++){

		//FIRST OF ALL, let's initialize the stream with the embeddings of the current token(s)
		if(wtype == WTYPE_FLOAT32){
			memcpy(&residualstream_layerstart[0][((b*T)+ppos)*dim_stream*wsize], &model->w.embeddings[((size_t)intok[b][pos + ppos]) * dim_stream * wsize], (dim_stream * wsize));
		}
		else
		if(wtype == WTYPE_BF16){
			for(size_t i = 0; i < dim_stream; i++){
				__bf16 emb_i = ((__bf16 *)&model->w.embeddings[((size_t)intok[b][pos + ppos]) * dim_stream * wsize])[i];
				((float *)&residualstream_layerstart[0][((b*T)+ppos)*dim_stream*sizeof(float)])[i] = (float)emb_i;
			}
		}
		else
		if(wtype == WTYPE_FLOAT16){
			for(size_t i = 0; i < dim_stream; i++){
				_Float16 emb_i = ((_Float16 *)&model->w.embeddings[((size_t)intok[b][pos + ppos]) * dim_stream * wsize])[i];
				((float *)&residualstream_layerstart[0][((b*T)+ppos)*dim_stream*sizeof(float)])[i] = (float)emb_i;
			}
		}
	    }
	  }
	}
	else{	//embeddings != NULL, starting embeddings have been provided by who's calling the forward() function

		memcpy(&residualstream_layerstart[0][0], embeddings, (dim_stream * sizeof(float) * B * T));
	}



#ifdef TRIP_DEBUG
wdebug(&residualstream_layerstart[0][dim_stream*sizeof(float)*0],WTYPE_FLOAT32,5,"initial stream @pos 0 from embeddings",-1,0);
#endif




		if( (model->config.architectures == ARCH_GEMMA_CAUSAL)
		    ||
		    ((model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL) && (model->config.submodel_type == MODELTYPE_DECODER))
		){

			float scale_factor = sqrtf((float)dim_stream);			

			multiply_vector(residualstream_layerstart[0], (dim_stream * B * T), scale_factor, residualstream_layerstart[0]);			

		}



	//THEN, let's apply absolute sinusoidal position encoding (Vaswani), if required
	if(	(model->config.pose_cfg == POSE_ORIGINAL)
		||
		(model->config.pose_cfg == POSE_LEARNED)
	){
	
		apply_positional_embeddings(model, residualstream_layerstart[0], residualstream_layerstart[0], dim_stream, pos, B, T);
	}



#ifdef TRIP_DEBUG
wdebug(&model->w.learned_pose_w[dim_stream*wsize*0],WTYPE_FLOAT32,5,"learned posei @pos 0",-1,0);
wdebug(&model->w.learned_pose_w[dim_stream*wsize*1],WTYPE_FLOAT32,5,"learned posei @pos 1",-1,0);
wdebug(&residualstream_layerstart[0][dim_stream*sizeof(float)*0],WTYPE_FLOAT32,5,"initial stream @pos 0 from embeddings after learned pose",-1,1);
#endif



		//LET'S FORWARD THE STREAM!, through all the layers of the model
		for(size_t layer = 0; layer < n_layers ; layer++){


	    		if(T > 1){
				if(layer == 0)  printf("\r\n");
				mylog(LOG_INFO,"parallel forwarding: starting layer %d of %d...",layer,n_layers);
			}


#ifdef TRIP_DEBUG
wdebug(stream,WTYPE_FLOAT32,12,"stream before PRE_ATT rmsnorm",layer,0);
wdebug(&stream[dim_stream*sizeof(float)*4],WTYPE_FLOAT32,12,"stream @pos 4 before PRE_ATT rmsnorm",layer,1);
#endif


			//PRE-attention normalization (if required)
			switch(model->config.norm_cfg[0]){

			case NORM_NONE:	
				memcpy(norm_pre_stream[layer], residualstream_layerstart[layer], (dim_stream*sizeof(float)*B*T));	
				break;	//do nothing
					
			case NORM_LAYERNORM: 		//LAYERNORM
				layernorm(norm_pre_stream[layer], residualstream_layerstart[layer], model->w.norm_pre_w[layer], model->w.norm_pre_b[layer], dim_stream, B, T,
					((action==ACTION_TRAIN) ? model->fm.norm_pre_mean[layer] : NULL), ((action==ACTION_TRAIN) ? model->fm.norm_pre_rstd[layer] : NULL) 
				);
				break;

			case NORM_RMSNORM:		//RMSNORM
				rmsnorm(model, norm_pre_stream[layer], residualstream_layerstart[layer], model->w.norm_pre_w[layer], dim_stream, B, T,
				        ((action==ACTION_TRAIN) ? model->fm.norm_pre_rrms[layer] : NULL)
				);
				break;

			default: 
				mylog(LOG_ERROR, "Invalid configuration 0x%02X for pre-attention normalization. Exiting...", model->config.norm_cfg[0]);
				exit(-1);
				break;
			}


#ifdef TRIP_DEBUG
wdebug(&bstream[dim_stream*sizeof(float)*4],WTYPE_FLOAT32,12,"bstream @pos 4 after PRE_ATT rmsnorm",layer,1);
#endif





			//now we cycle over all the attention heads (same number of the queries, in multi-head, multi-query, and grouped-query attention schemes)

			size_t last_qhead  = -1;	//yes, it's -1 to unsigned; I just want to put a bad value in it
			size_t last_kvhead = -1;

			for(size_t head = 0; (head < n_queries); head++){

				//generalization for each possible case at the best of my knowledge:
				// - "multi-head"	(number of queries = number of keys&values)	
				// - "multi-query"	(number of keys&values = 1)
				// - "grouped-query"	(number of keys and values = sub-multiple for number of queries)

				size_t kv = ((head * n_keys)  /  n_queries);





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






	//COMPUTATION of the QUERY VECTOR

	size_t q_offset;

	byte * q;

	if(last_qhead != head){

	    for(b = 0; b < B; b++){

		q_offset  = 0;
		q_offset += head;			//currently, we are processing query #head (over n_queries in total)
		q_offset *= B;				//each query area in the cache is a sub-block, made up of "B" sequences
		q_offset += b;				//currently, we are processing sequence #b (over B sequences in total)
		q_offset *= T;				//each sequence/query area in the forward memory is a sub-sub-block, made up of "T" positions
		//q_offset += pos;			//currently, we are processing position #pos (over sequence_maxtokens in total)
		q_offset *= dim_qkv;			//each pos/sequence/query in the cache is a query vector, made up of "dim_qkv" components
		//since the cache is actually "float *", we need to multiply by the size of float32
		q_offset *= sizeof(float);

		q = &model->fm.queries[layer][q_offset];


		if((checkpoint_type==CP_SAFETENSORS)  && (model->config.pose_cfg!=POSE_LEARNED)  ){
		  matmulf_nt_interleaved( &model->w.qm[layer][qom_offset], &norm_pre_stream[layer][((b*T)+0)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), &q[0]);
		}
		else{
		  matmulf_nt( &model->w.qm[layer][qom_offset], &norm_pre_stream[layer][((b*T)+0)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), &q[0]);
		}


		if(model->config.bias_cfg[3] == BIAS_ON){
		
			#pragma omp parallel for private(ppos)
			for(ppos = 0; ppos < T; ppos++){

		  		sum_vectors(
					&q[ppos*dim_qkv*sizeof(float)], WTYPE_FLOAT32,
					&model->w.qb[layer][head*dim_qkv*wsize], wtype,
					dim_qkv, 
					&q[ppos*dim_qkv*sizeof(float)]
				);
			}
		}


//if(layer==(n_layers-1))
#ifdef TRIP_DEBUG
wdebug(q, WTYPE_FLOAT32, 5, "q", -1, 1);
#endif

//if(head==1) wdebug(&q[dim_qkv*sizeof(float)*4], WTYPE_FLOAT32, dim_qkv, "q @pos 4 layer 0 head 1 before ROPE", -1, 1);


		//let's apply RoPE positional embeddings, if this is the case
		if(model->config.pose_cfg == POSE_ROPE){

			if(last_qhead != head)	apply_positional_embeddings(model, &q[0], &q[0], dim_qkv, pos, 1, T);
		}
	    }
	}



	//3) offset of the current key vector (current layer, current head's key, current position of the sequence) in the model->fm.keys;
	//   the offset is the same for the value vector
	
	//IMPORTANT NOTE about key cache and value cache:
	//at each layer, for each head, we put all the keys (from position <zero> to <maxseq-1>) side by side, to allow matrix multiplications in the attention computation
	
	size_t kv_offset;

	byte * k;
	byte * v;






	//COMPUTATION of the KEY VECTOR
	// there's one for each position, calculated at each time-step;
	// this is exactly the reason we keep a cache: 	
	// to avoid repeating the computation every time for each time-step "pos".	
	// Now we need the key and value vectors for the first time,
	// thus we are CALCULATING them and STORING them to the cache.
	// This is the KEY vector for this position in the sequence, for this head's key, at this layer.
	if(last_kvhead != kv){

	    for(b = 0; b < B; b++){

		kv_offset  = 0;
		kv_offset += kv;				//currently, we are processing key #kv (over n_keys in total)
		kv_offset *= B;				//each key area in the cache is a sub-block, made up of "B" sequences
		kv_offset += b;				//currently, we are processing sequence #b (over B sequences in total)
		kv_offset *= max_tokens;			//each sequence/key area in the cache is a sub-sub-block, made up of "sequence_maxtokens" positions
		kv_offset += pos;				//currently, we are processing position #pos (over sequence_maxtokens in total)
		kv_offset *= dim_qkv;			//each pos/sequence/key in the cache is a key vector, made up of "dim_qkv" components
		//since the cache is actually "float *", we need to multiply by the size of float32
		kv_offset *= sizeof(float);

		k = &model->fm.keys[layer][kv_offset];


		if((checkpoint_type==CP_SAFETENSORS)  && (model->config.pose_cfg!=POSE_LEARNED)  ){
			matmulf_nt_interleaved( &model->w.km[layer][kvm_offset], &norm_pre_stream[layer][((b*T)+0)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), &k[0]);
		}
		else{
			matmulf_nt(             &model->w.km[layer][kvm_offset], &norm_pre_stream[layer][((b*T)+0)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), &k[0]);
		}

		if(model->config.bias_cfg[3] == BIAS_ON){

			#pragma omp parallel for private(ppos) 
			for(ppos = 0; ppos < T; ppos++){
		  		sum_vectors(
					&k[ppos*dim_qkv*sizeof(float)], WTYPE_FLOAT32,
					&model->w.kb[layer][head*dim_qkv*wsize], wtype,
					dim_qkv, 
					&k[ppos*dim_qkv*sizeof(float)]
				);
			}
		}


		//let's apply RoPE positional embeddings, if this is the case
		if(model->config.pose_cfg == POSE_ROPE){
	
//if(kv==1) wdebug(&k[dim_qkv*sizeof(float)*4], WTYPE_FLOAT32, dim_qkv, "k @pos 4 layer 0 kv_head 1 before ROPE", -1, 1);

			//given the structure of the kv memory, I need to apply pose to one sequence at a time:  1,T   instead of B,T
			if(last_kvhead != kv)	apply_positional_embeddings(model, k, k, dim_qkv, pos, 1, T);


		}

//if(kv==1) wdebug(&k[dim_qkv*sizeof(float)*4], WTYPE_FLOAT32, dim_qkv, "k @pos 4 layer 0 kv_head 1 AFTER ROPE", -1, 1);

	    }


	}



	//COMPUTATION of the VALUE VECTOR
	if(last_kvhead != kv){


	    for(b = 0; b < B; b++){

		kv_offset  = 0;
		kv_offset += kv;				//currently, we are processing key #kv (over n_keys in total)
		kv_offset *= B;				//each key area in the cache is a sub-block, made up of "B" sequences
		kv_offset += b;				//currently, we are processing sequence #b (over B sequences in total)
		kv_offset *= max_tokens;			//each sequence/key area in the cache is a sub-sub-block, made up of "sequence_maxtokens" positions
		kv_offset += pos;				//currently, we are processing position #pos (over sequence_maxtokens in total)
		kv_offset *= dim_qkv;			//each pos/sequence/key in the cache is a key vector, made up of "dim_qkv" components
		//since the cache is actually "float *", we need to multiply by the size of float32
		kv_offset *= sizeof(float);

		v = &model->fm.values[layer][kv_offset];


		matmulf_nt( &model->w.vm[layer][kvm_offset], &norm_pre_stream[layer][((b*T)+0)*dim_stream*sizeof(float)], dim_stream,dim_qkv, dim_stream,(1*T), &v[0]);

		if(model->config.bias_cfg[3] == BIAS_ON){

			#pragma omp parallel for private(ppos)
			for(ppos = 0; ppos < T; ppos++){
		  		sum_vectors(
					&v[ppos*dim_qkv*sizeof(float)], WTYPE_FLOAT32,
					&model->w.vb[layer][head*dim_qkv*wsize], wtype,
					dim_qkv, 
					&v[ppos*dim_qkv*sizeof(float)]
				);
			}
		}

//if(kv==1) wdebug(&v[dim_qkv*sizeof(float)*4], WTYPE_FLOAT32, dim_qkv, "v @pos 4 layer 0 kv_head 1 (AFTER ROPE on q&k)", -1, 1);

	    }


	}



#ifdef TRIP_DEBUG
wdebug(q, WTYPE_FLOAT32, dim_qkv, "q layer 0 head 0 before attention", -1, 1);
#endif



	/*
	//let's apply RoPE positional embeddings, if this is the case

	if(model->config.pose_cfg == POSE_ROPE){
	
		if(last_qhead != head)	apply_positional_embeddings(model, q, q, dim_qkv, pos, B, T);
		if(last_kvhead != kv)	apply_positional_embeddings(model, k, k, dim_qkv, pos, B, T);
	}
	*/






	//NOW, to prepare the attention computation, both for keys and values we need to point to the cache for this layer/head
	//from sequence ZERO, pos ZERO, and NOT from the current position

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



	//we need to reset q to the proper offset
	q_offset  = 0;
	q_offset += head;			//currently, we are processing query #head (over n_queries in total)
	q_offset *= B;				//each query area in the cache is a sub-block, made up of "B" sequences
	//q_offset += b;				//currently, we are processing sequence #b (over B sequences in total)
	q_offset *= T;				//each sequence/query area in the forward memory is a sub-sub-block, made up of "T" positions
	//q_offset += pos;			//currently, we are processing position #pos (over sequence_maxtokens in total)
	q_offset *= dim_qkv;			//each pos/sequence/query in the cache is a query vector, made up of "dim_qkv" components
	//since the memory is actually "float *", we need to multiply by the size of float32
	q_offset *= sizeof(float);

	q = &model->fm.queries[layer][q_offset];
	


	size_t scores_offset = head * (sizeof(float) * T * T * B);

	//ATTENTION!!! :)
	attention_head_io( 
		model, &model->fm.raw_attention_scores[layer][scores_offset], &model->fm.attention_scores[layer][scores_offset],
		&heads_output[layer][0+(head*dim_qkv*sizeof(float))], &q[0], pos, ks, vs, 0, ((pos-0)+T), 
		attention_type, B, T, intok
	);	//it's (pos+1) since "pos" starts from 0,



//if(head==0) wdebug(&heads_output[((dim_stream*4)+(head*dim_qkv))*sizeof(float)], WTYPE_FLOAT32, dim_qkv, "attention head output @pos 4 layer 0 head 0", -1, 1);



				last_qhead = head;
				last_kvhead = kv;
	

			}	//for( head < n_queries )	






#ifdef TRIP_DEBUG
wdebug(&heads_output[dim_stream*sizeof(float)*4], WTYPE_FLOAT32, dim_stream, "full attention values before output proj  @pos 4 at layer 0", -1, 1);
#endif
//if(pos==1)	exit(1);





			//PROJECTION of the outputs of all the heads to their final mix, which we will then immediately add to the main residual stream (RESIDUAL CONNECTION)
 
			//NOTE: we could do this SEPARATELY for each head's output, and sum the contribution from each one, but the lines of the output weights matrix for each head 
			//are interleaved with the lines of the other heads, at least in the usual format in which this matrix is stored in the AI world, so... 
			//it's much easier to do this all in once.


			matmulf_nt( model->w.om[layer], heads_output[layer], dim_stream,dim_stream, dim_stream,(B*T), attentionlayer_out_stream[layer]);	//attentionlayer_out_stream([B*T])[dim_stream]

			if(model->config.bias_cfg[3] == BIAS_ON){

				#pragma omp parallel for collapse(2)
				for(b = 0; b < B; b++){
				    for(ppos = 0; ppos < T; ppos++){

					sum_vectors(
						&attentionlayer_out_stream[layer][dim_stream*sizeof(float)*((b*T)+ppos)], WTYPE_FLOAT32,
						&model->w.ob[layer][0], wtype, 
						dim_stream, 
						&attentionlayer_out_stream[layer][dim_stream*sizeof(float)*((b*T)+ppos)]
					);
				    }
				}
			}
	


#ifdef TRIP_DEBUG
wdebug(&attentionlayer_out_stream[dim_stream*sizeof(float)*1], WTYPE_FLOAT32, dim_stream, "global attention output stream @pos 1 at layer 0", -1, 1);
#endif




			//RESIDUAL CONNECTION
			//notice that we sum up the output stream of the attention layer 
			//NOT to bstream (normalized), 
			//BUT to stream (which is like bstream, but before pre-attention normalization)
			sum_vectors(attentionlayer_out_stream[layer], WTYPE_FLOAT32, residualstream_layerstart[layer], WTYPE_FLOAT32, (dim_stream*B*T), residualstream_after_attention[layer]);
		

	
#ifdef TRIP_DEBUG
wdebug(stream,WTYPE_FLOAT32,12,"stream BEFORE POST_ATT norm",layer,1);
#endif

			//POST-attention normalization (pre-Feed Forward Network)   if required
			
			switch(model->config.norm_cfg[1]){

			case NORM_NONE:
				memcpy(norm_post_stream[layer], residualstream_after_attention[layer], (dim_stream*sizeof(float)*B*T));	
				break;	//do nothing
					
			case NORM_LAYERNORM: 	//LAYERNORM
				layernorm(norm_post_stream[layer], residualstream_after_attention[layer], model->w.norm_post_w[layer], model->w.norm_post_b[layer], dim_stream, B, T,
				        ((action==ACTION_TRAIN) ? model->fm.norm_post_mean[layer] : NULL), ((action==ACTION_TRAIN) ? model->fm.norm_post_rstd[layer] : NULL) 
				);
				break;

			case NORM_RMSNORM:	//RMSNORM
				rmsnorm(model, norm_post_stream[layer], residualstream_after_attention[layer], model->w.norm_post_w[layer], dim_stream, B, T,
				        ((action==ACTION_TRAIN) ? model->fm.norm_post_rrms[layer] : NULL)
				);
				break;

			default: 
				mylog(LOG_ERROR, "Invalid configuration 0x%02X for post-attention normalization. Exiting...", model->config.norm_cfg[1]);
				exit(-1);
				break;
			}


			//LINEAR LAYER  +  Feed Forward Network  +  LINEAR LAYER


			
			//LINEAR LAYER   BEFORE   the Feed Forward Network

		
			matmulf_nt(model->w.pre_ffn_w[layer], norm_post_stream[layer], dim_stream,hidden_dim, dim_stream,(B*T), ffn_in_stream[layer]);


			if(model->config.bias_cfg[0] == BIAS_ON){

				#pragma omp parallel for collapse(2)
				for(b = 0; b < B; b++){
				    for(ppos = 0; ppos < T; ppos++){

					sum_vectors(
						&ffn_in_stream[layer][((b*T)+ppos)*hidden_dim*sizeof(float)], WTYPE_FLOAT32, 
						model->w.pre_ffn_b[layer], wtype, 
						hidden_dim, 
						&ffn_in_stream[layer][((b*T)+ppos)*hidden_dim*sizeof(float)]
					);
				    }
				}
			}




			//this is the management of a special case: gated non-linearity, like LLAMA2 SILU, which requires additional multipliers
			int ffn_gating;

			if((model->config.ffn_nl_type[1] == GATE_ON)  ||  (model->config.ffn_nl_type[0] == NL_SILU_LLAMA))	ffn_gating = 1;
			else													ffn_gating = 0;
			


			if(ffn_gating == 1){
				
				matmulf_nt(model->w.pre_ffn_w2[layer], norm_post_stream[layer], dim_stream,hidden_dim, dim_stream,(B*T), ffn_aux_stream[layer]);
			}



			#pragma omp parallel for collapse(2)
			for(b = 0; b < B; b++){			
			    for(ppos = 0; ppos < T; ppos++){
			
				//FEED FORWARD NETWORK

				ffn_io(	model->config.ffn_nl_type[0],
					&ffn_out_stream[layer][((b*T)+ppos)*hidden_dim*sizeof(float)], 
					&ffn_in_stream[layer][ ((b*T)+ppos)*hidden_dim*sizeof(float)], 
					((ffn_gating == 1) ? &ffn_aux_stream[layer][((b*T)+ppos)*hidden_dim*sizeof(float)] : NULL), 
					hidden_dim
				);
			    }
			}



			//LINEAR LAYER   AFTER   the Feed Forward Network

			matmulf_nt(model->w.post_ffn_w[layer], ffn_out_stream[layer], hidden_dim,dim_stream, hidden_dim,(B*T), ffn_final_stream[layer]);


			if(model->config.bias_cfg[1] == BIAS_ON){

			    #pragma omp parallel for collapse(2)
			    for(b = 0; b < B; b++){
			    	for(ppos = 0; ppos < T; ppos++){

					sum_vectors(
						&ffn_final_stream[layer][((b*T)+ppos)*dim_stream*sizeof(float)], WTYPE_FLOAT32, 
						model->w.post_ffn_b[layer], wtype, 
						dim_stream, 
						&ffn_final_stream[layer][((b*T)+ppos)*dim_stream*sizeof(float)]
					);
				}
			    }
			}

	
		
			
			//RESIDUAL CONNECTION (let's sum the output stream of the processing of the Feed Forward Network to the original input stream)
			//notice that, just like after the attention layer, we sum up the output stream of the FFN layer 
			//NOT to bstream (normalized), 
			//BUT to stream (which is like bstream, but before pre-FFN normalization)
	
			sum_vectors(residualstream_after_attention[layer], WTYPE_FLOAT32, ffn_final_stream[layer], WTYPE_FLOAT32, (dim_stream*B*T), residualstream_after_ffn[layer]);


//if((pos==1)) wdebug(stream, WTYPE_FLOAT32, 5, "stream at end of layer %d", layer, 1);
#ifdef TRIP_DEBUG
wdebug(&stream[dim_stream*sizeof(float)*4], WTYPE_FLOAT32, 12, "stream @pos 4 at end of layer %d", layer, 1);
#endif
	
				
			if(	(action == ACTION_TRAIN)
				&&
				(runtime_actions & (1<<SIGUSR2))
			){
				return ppos;
			}

	
		}	//for( layer < n_layers )




#ifdef TRIP_DEBUG
wdebug(stream, WTYPE_FLOAT32, 5, "stream before final rmsnorm", -1, 0);
#endif

	//if we were doing parallel decoding of the user input, we are done with the parallel processing
	//let's align everything to the final step
	if(	((action == ACTION_DECODE) || (action == ACTION_CHAT))
		&&
		(T != 1)
	){

		if(chat_flags & CHAT_SAVE_CONTEXT){

			chat_save_context(model, intok[0], &n_intoks[0]);	//intok[0]: we assume that we want to save only the context of the first sequence

			mylog(LOG_INFO,"Chat context file \"%s\" successfully saved. Now exiting...", chat_context_file);
			exit(1);
		}

		if(calculate_loss == false){

			pos = pos + T - 1;

			for(b = 0; b < B; b++){

				//memcpy(&stream[((b*T)+0)*dim_stream*sizeof(float)], &stream[((b*T) + (T-1))*dim_stream*sizeof(float)], (dim_stream*sizeof(float)));
				//                ^^^^^
				//                ^^^^^ 
				//NO! We need to compact all the last stream vectors of each batch, because we still need to matmul them to get the logits
				//              
				memcpy(	&residualstream_after_ffn[n_layers-1][  b            *dim_stream*sizeof(float)], 
					&residualstream_after_ffn[n_layers-1][((b*T) + (T-1))*dim_stream*sizeof(float)], 
					(dim_stream*sizeof(float))
				);

			}

			T = 1;
		}

	}
	else
	//if we were doing parallel DECODING, we are done with the parallel processing - we will just need to apply the final norm, and then exit 
	if(	(action == ACTION_VISION)
		&&
		(model->config.submodel_type == MODELTYPE_DECODER)
		&&
		(attention_type == ATTENTION_FULL)
		&&
		(T != 1)
	){

/*
		if(chat_flags & CHAT_SAVE_CONTEXT){

			chat_save_context(model, intok[0], &n_intoks[0]);	//intok[0]: we assume that we want to save only the context of the first sequence	

			mylog(LOG_INFO,"Chat context file \"%s\" successfully saved. Now exiting...", chat_context_file);
			exit(1);
		}
*/

		pos = pos + T - 1;
	
		for(b = 0; b < B; b++){

			//memcpy(&stream[((b*T)+0)*dim_stream*sizeof(float)], &stream[((b*T) + (T-1))*dim_stream*sizeof(float)], (dim_stream*sizeof(float)));
			//                ^^^^^
			//                ^^^^^ 
			//NO! We need to compact all the last stream vectors of each batch, because we still need to matmul them to get the logits

			memcpy(	&residualstream_after_ffn[n_layers-1][  b            *dim_stream*sizeof(float)], 
				&residualstream_after_ffn[n_layers-1][((b*T) + (T-1))*dim_stream*sizeof(float)], 
				(dim_stream*sizeof(float))
			);


		}	

		T = 1;

		
		//from now on, attention must switch to CAUSAL
		attention_type = ATTENTION_CAUSAL;


	}
	else
	//if we were doing parallel encoding, we are done with the parallel processing - we will just need to apply the final norm, and then exit 
	if(	(model->config.submodel_type == MODELTYPE_VISION_ENCODER)
		&&
		(T != 1)
	){


		//do nothing!
	}




	//FINAL normalization (output layer, right after the last Attention+FFN layer)
	//size_t startpos = pos;
	//size_t endpos   = pos + T;

			
	    switch(model->config.norm_cfg[2]){

		case NORM_NONE:
			memcpy(norm_final_stream, residualstream_after_ffn[n_layers-1], (dim_stream * B * T * sizeof(float)));
		break;
					
		case NORM_LAYERNORM: 	//LAYERNORM
			layernorm(norm_final_stream, residualstream_after_ffn[n_layers-1], model->w.norm_final_w, model->w.norm_final_b, dim_stream, B, T,
			        ((action==ACTION_TRAIN) ? model->fm.norm_final_mean : NULL), ((action==ACTION_TRAIN) ? model->fm.norm_final_rstd : NULL) 
			);
			//layernorm(&stream[(ppos-startpos)*dim_stream*sizeof(float)], &stream[(ppos-startpos)*dim_stream*sizeof(float)], &model->w.norm_final_w[0], &model->w.norm_final_b[0], dim_stream);
		break;

		case NORM_RMSNORM:	//RMSNORM
			rmsnorm(model, norm_final_stream, residualstream_after_ffn[n_layers-1], model->w.norm_final_w, dim_stream, B, T,
				((action==ACTION_TRAIN) ? model->fm.norm_final_rrms : NULL)	
			);
			//rmsnorm(model, &stream[(ppos-startpos)*dim_stream*sizeof(float)], &stream[(ppos-startpos)*dim_stream*sizeof(float)], &model->w.norm_final_w[0], dim_stream);
			break;

		default: 
			mylog(LOG_ERROR, "Invalid configuration 0x%02X for FINAL normalization. Exiting...", model->config.norm_cfg[2]);
			exit(-1);
			break;
	    }



#ifdef TRIP_DEBUG
wdebug(stream, WTYPE_FLOAT32, 5, "stream after final rmsnorm", -1, 0);
#endif


	    if(out_streams != NULL){

		memcpy(&out_streams[0], norm_final_stream, (B * T * dim_stream * sizeof(float)));

		break;	//let's exit the forwarding loop

	    }
	    else{


	      	size_t startpos = pos;
	      	size_t endpos   = pos + T;


		//CLASSIFICATION of the output stream TO a vector of probabilities all over the possibile entries in the vocabulary
		matmulf_nt(model->w.logits_classifier, norm_final_stream, dim_stream,vocab_size, dim_stream,(B*T), logits);



#ifdef TRIP_DEBUG
wdebug(logits, WTYPE_FLOAT32, 5, "final logits", -1, 0);
#endif

	float * sequence_loss;

	if(calculate_loss == true){

		sequence_loss = myalloc(sizeof(float) * B);
	}


	for(b = 0; b < B; b++){

		pos = startpos;

	      while( pos < endpos ){

		//now, what to do with the output distribution of probabilities?
		
		//If we are still WITHIN the range of the input sequence of tokens,
		//all the processing above was just to calculate the keys and values for this position in the sequence for each layer; 
		//they will be used for the attention computations in all the following positions of the sequence;
		//so we can:
		//1) just skip the sampling (i.e.: selection) of the next token (since it will be overwritten by the actual next token specified in the input sequence)
		//   OR
		//2) we can still sample the next token; this is required in the case of the training of the model, or for test purposes
		//
		//
		//If we are BEYOND the range of the input sequence of tokens,
		//then we MUST sample the next token; 
		//otherwise, we will have no input for the first layer in the next step (position of the sequence)


	
		int next_token;
		float out_prob;
		next_token = sample_next_token(model, &logits[((b*T) + (pos-startpos))*(sizeof(float)*vocab_size)], &out_prob);


		char report_prefix[256];

		if(B>1){	
			//if this is a batch, and we are processing one of the sequences in the batch:
			sprintf(report_prefix,"batch_seq %.5zd  ",b);
		}	
		else{
			report_prefix[0] = '\0';
		}


		sprintf(lbuf,"%spos %.5zd:\t\t\t\tnext_token = %.5d\t%20s", report_prefix, pos, next_token, (next_token>=0)?toki.vocab[next_token]:"(error)");



		pos++;



		//if we are generating brand new tokens	
		//(during training, this may happen if we are testing the decode every many training steps, as a sample of the current quality of the model)
		if( (action != ACTION_TRAIN)
		    &&
		    ((pos >= n_intoks[b])  &&  (pos < max_tokens))
		){


			n_intoks[b]++;

			intok[b][pos] = next_token;

			if(intok[b][pos] == toki.eos_id){	//if we generate EOS token, decode has finished 

			    	//Reset all formatting at the end
				chat_textformat_reset();
 			
				mylog(LOG_DEBUG,lbuf);
				
	    			T = 1;

				free_forward_memory(model);

				return pos;
				
			    break; 
			}

		}
		else
		if((calculate_loss == true)  &&  (pos < n_intoks[b])){

			float target_prob = 1.0;
			float actual_prob = ((float *)&(logits[((b*T) + ((pos-1)-startpos))*(sizeof(float)*vocab_size)]))[intok[b][pos]];	//this is the calculated probability for the actual next token (not the predicted one)
			float loss;
					
			crossentropy(&actual_prob, &target_prob, 1, &loss);

			sprintf(endstr(lbuf),"\t\tLOSS = %.5f  to actual next token \"%s\" (p=%.5f)", loss, toki.vocab[intok[b][pos]], actual_prob);

			sequence_loss[b] += loss;
		}
		else{

			lbuf[0] = '\0';
		}



		if(strlen(lbuf) > 0)	mylog(LOG_DEBUG,lbuf);

	      }	//while( pos < endpos )

	}	//loop over sequences in the batch


/*
wdebug(&logits[((0*T)+0)*vocab_size*sizeof(float)], WTYPE_FLOAT32, 5, "logits[((0*T)+0)*vocab_size]",-1,0);
wdebug(&logits[((0*T)+1)*vocab_size*sizeof(float)], WTYPE_FLOAT32, 5, "logits[((0*T)+1)*vocab_size]",-1,0);
wdebug(&logits[((1*T)+1)*vocab_size*sizeof(float)], WTYPE_FLOAT32, 5, "logits[((1*T)+1)*vocab_size]",-1,0);
*/



	if(calculate_loss == true){

		for(b=0;b<B;b++){
			
			mylog(LOG_INFO, "sequence #%d mean loss = %.3f", b, (sequence_loss[b] / ((float)n_intoks[b])) );
		}

	/*
		char buf[16];
		sprintf(buf,"%f",sequence_loss[0]);
		if(strcmp(buf,"nan")==0)
		{
			printf("\r\n\r\nbuf=\"%s\"\r\n\r\n",buf);
			exit(1);
		}
	*/

		save_loss(sequence_loss, n_intoks, B);	

		free(sequence_loss);
	}



	    }//"if(out_streams==NULL)"



	    T = 1;



	}	//for( pos < max_tokens )
		
		


	if(action != ACTION_TRAIN){

		//let's free up the dynamically-allocated memory before we leave
		free_forward_memory(model);
	}


	return pos;
}



