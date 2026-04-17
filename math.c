#define TRIP_MATH_VERSION 2026041501
#include "trip.h"


// ============================================================
//  Attention head I/O
//
//  This is somehow a "context information multiplexer":
//  - query, key, and value vectors are computed as a projection 
//    of the normalized residual stream. Keys and values come also
//    from the past (i.e.: previous timesteps, previous tokens)
//    when attention is "causal" (i.e.: respectful of temporal order),
//    and also from following positions when attention is "full".
//  - given a query ("what pattern am I looking for?"), 
//    it scores every key in the sequence ("how relevant is each position?"), 
//    the key being "what is the relevant data pattern of this position 
//    at this layer for this attention head?"
//  - it normalizes those scores with softmax, and uses them to
//    blend the value vectors into a single output; value vectors
//    being "what information do I want to output from this
//    position at this layer for this attention head?".
//
//  Causal masking ensures that a token can only attend to
//  positions before it. Full attention looks at all positions.
//
//  The backward pass computes gradients for Q, K, V, and the
//  attention scores themselves (the chain rule through softmax
//  and the score-value product).
// ============================================================




// head_out = pointer to the output vector of this head; NOTE: the output vector is [q_npos][heads][dim_qkv]
// queries = pointer to the vector of query vectors
// keys = pointer to the keys cache
// values = pointer to the values cache
//

//q_startpos  is the [position in the sequence of tokens] corresponding to the vector in first position in _queries
//kv_startpos is the [position in the sequence of tokens] corresponding to the vector in first position in _keys and _values

int attention_head_io(Model * model, byte * _raw_attention_scores, byte * _attention_scores, byte * _head_out, byte * _queries, size_t q_startpos, byte * _keys, byte * _values, size_t kv_startpos, size_t kv_npos, int attention_type, size_t B, size_t T, int ** intok){

	size_t b,t;
	size_t q_npos = T;	//alias, just to make it more clear that T here means "how many query vectors/positions"

	size_t max_tokens = model->config.sequence_maxtokens;
	size_t dim_stream = model->config.dim_stream;
	size_t dim_qkv = model->config.dim_stream / model->config.n_queries;

	if(action == ACTION_TRAIN){

		max_tokens = T;
	}



	float inv_div_k = 1.0 / sqrtf((float)dim_qkv);


	//all calculations here require float32 for maximum precision
	int bak_wtype = wtype;
	wtype = WTYPE_FLOAT32;
	wsize = wtype_bytesize[wtype];	


    byte * _raw_attention_scores_base = _raw_attention_scores;
    byte * _attention_scores_base = _attention_scores;

    byte * _head_out_base = _head_out;
    byte * _queries_base  = _queries;
    byte * _keys_base     = _keys;
    byte * _values_base   = _values;

    for(size_t b = 0; b < B; b++){

	_raw_attention_scores = _raw_attention_scores_base + (b * (q_npos * kv_npos) * sizeof(float));
	_attention_scores     = _attention_scores_base     + (b * (q_npos * kv_npos) * sizeof(float));

	_head_out = _head_out_base + (b * dim_stream *     q_npos * sizeof(float));
	_queries  = _queries_base  + (b * dim_qkv    *     q_npos * sizeof(float));
	_keys     = _keys_base     + (b * dim_qkv    * max_tokens * sizeof(float));
	_values   = _values_base   + (b * dim_qkv    * max_tokens * sizeof(float));



	if(attention_type == ATTENTION_FULL){

	   matmulf_nt(_keys, _queries, dim_qkv, kv_npos, dim_qkv, q_npos, _raw_attention_scores);	//raw_attention_scores[q_npos][kv_npos]

	   multiply_vector(_raw_attention_scores, (q_npos * kv_npos), inv_div_k, _raw_attention_scores);



	   #pragma omp parallel for
	   for(size_t i=0; i<q_npos; i++){

		byte * _w_value          = myalloc(dim_qkv * sizeof(float));

		float * raw_attention_scores_qpos = (float *)&_raw_attention_scores[i*kv_npos*sizeof(float)];


		if((intok != NULL) && (action != ACTION_VISION)){
		   	for(size_t ii=0; ii<kv_npos; ii++){
				if(intok[b][kv_startpos + ii] == toki.pad_id){
					raw_attention_scores_qpos[ii] = -INFINITY;
				}
			}
		}


		softmax(&_raw_attention_scores[i*kv_npos*sizeof(float)], 1, kv_npos, &_attention_scores[i*kv_npos*sizeof(float)]);	//attention_scores_softmaxed_row-wise[q_npos][kv_npos]

//it would be:
//matmulf_nn(&_attention_scores[i*kv_npos*sizeof(float)], _values, kv_npos, 1, dim_qkv, kv_npos, &_head_out[i*dim_stream*sizeof(float)]);   //head_out[q_npos][heads][dim_qkv]

		//set to zero this head_out vector
		multiply_vector(&_head_out[(i*dim_stream)*sizeof(float)], dim_qkv, 0.0, &_head_out[(i*dim_stream)*sizeof(float)]);

		for(size_t j = 0; j < kv_npos; j++){
			float w = *(float *)&_attention_scores[((i*kv_npos)+j)*sizeof(float)];
			multiply_vector(&_values[(j*dim_qkv)*sizeof(float)], dim_qkv, w, _w_value);
			sum_vectors(&_head_out[(i*dim_stream)*sizeof(float)], WTYPE_FLOAT32, _w_value, WTYPE_FLOAT32, dim_qkv, &_head_out[(i*dim_stream)*sizeof(float)]);		
		}

		free(_w_value);
	   }


	}
	else
	if(attention_type == ATTENTION_CAUSAL){

	   #pragma omp parallel for
	   for(size_t pos = q_startpos; pos < (q_startpos+q_npos); pos++){

	      if(pos>=kv_startpos){

		 byte * _w_value          = myalloc(dim_qkv * sizeof(float));

	         //matmulf_nt_fullf32
	         matmulf_nt
		 (
			&_queries[(pos-q_startpos)*dim_qkv*sizeof(float)], _keys, 
			dim_qkv,1, 
			dim_qkv,(pos-kv_startpos+1), 
			&_raw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)]
	         );      //raw_attention_scores [][pos-kv_startpos+1]


	         multiply_vector(&_raw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], (pos-kv_startpos+1), inv_div_k, &_raw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)]);
		 float * raw_attention_scores_qpos = (float *)&_raw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)];

	   	 for(size_t ii=0; ii<kv_npos; ii++){

			if((kv_startpos+ii) > pos){	//this is CAUSAL ATTENTION!
				raw_attention_scores_qpos[ii] = -INFINITY;
			}
			if((intok[b][kv_startpos + ii] == toki.pad_id)  &&  (action != ACTION_VISION)){
				raw_attention_scores_qpos[ii] = -INFINITY;
			}
		 }

		
	         softmax(
			&_raw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], 
			1,(pos-kv_startpos+1), 
			&_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)]
	         );	//attention_scores softmaxed row-wise[ ][kv_npos], reducing computation to the non-zero portion of the attention scores vector

		 //this should not be necessary...
		 memset(&_attention_scores[(((pos-q_startpos)*kv_npos) + (pos-kv_startpos+1))*sizeof(float)],  0  ,((kv_npos-(pos-kv_startpos+1))*sizeof(float)));


//it would be:
//matmulf_nn(&_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], _values, kv_npos, 1, dim_qkv, kv_npos, &_head_out[(pos-q_startpos)*dim_stream*sizeof(float)]);	//head_out[q_npos][heads][dim_qkv]


	    	//set to zero this head_out vector
		multiply_vector(&_head_out[((pos-q_startpos)*dim_stream)*sizeof(float)], dim_qkv, 0.0, &_head_out[((pos-q_startpos)*dim_stream)*sizeof(float)]);

		for(size_t j = 0; j < kv_npos; j++){
			float w = *(float *)&_attention_scores[(((pos-q_startpos)*kv_npos)+j)*sizeof(float)];
			multiply_vector(&_values[(j*dim_qkv)*sizeof(float)], dim_qkv, w, _w_value);
			sum_vectors(&_head_out[((pos-q_startpos)*dim_stream)*sizeof(float)], WTYPE_FLOAT32, _w_value, WTYPE_FLOAT32, dim_qkv, &_head_out[((pos-q_startpos)*dim_stream)*sizeof(float)]);		
		}

		free(_w_value);
  	      }

	   }

	}

    }	//loop over the batches



	wtype = bak_wtype;	//let's restore the general data type
	wsize = wtype_bytesize[wtype];	


	//free(_attention_scores);

	return 0;
}


int attention_head_io_backward(
	byte * _dqueries, byte * _dkeys, byte * _dvalues,
	byte * _queries, byte * _keys, byte * _values, size_t q_startpos, size_t kv_startpos, size_t kv_npos, byte * _dhead_out,
	byte * _attention_scores, byte * _raw_attention_scores, byte * _dattention_scores, byte * _draw_attention_scores, 
	Model * model, int attention_type, size_t B, size_t T
//	Model * model, byte * _head_out, byte * _queries, size_t q_startpos, byte * _keys, byte * _values, size_t kv_startpos, size_t kv_npos, int attention_type, size_t B, size_t T
){

	size_t b,t;
	size_t q_npos = T;	//alias, just to make it more clear that T here means "how many query vectors/positions"

	//size_t max_tokens = model->config.sequence_maxtokens;
	size_t max_tokens = T;

	size_t dim_stream = model->config.dim_stream;
	size_t dim_qkv = model->config.dim_stream / model->config.n_queries;

	


	float inv_div_k = 1.0 / sqrtf((float)dim_qkv);


	//all calculations here require float32 for maximum precision
	int bak_wtype = wtype;
	wtype = WTYPE_FLOAT32;
	wsize = wtype_bytesize[wtype];	



    byte * _raw_attention_scores_base = _raw_attention_scores;
    byte * _attention_scores_base     = _attention_scores;

    byte * _draw_attention_scores_base = _draw_attention_scores;
    byte * _dattention_scores_base     = _dattention_scores;

    //byte * _head_out_base = _head_out;
    byte * _queries_base  = _queries;
    byte * _keys_base     = _keys;
    byte * _values_base   = _values;

    byte * _dhead_out_base = _dhead_out;
    byte * _dqueries_base  = _dqueries;
    byte * _dkeys_base     = _dkeys;
    byte * _dvalues_base   = _dvalues;



    for(size_t b = 0; b < B; b++){

	_raw_attention_scores = _raw_attention_scores_base + (b * (q_npos * kv_npos) * sizeof(float));
	_attention_scores     = _attention_scores_base     + (b * (q_npos * kv_npos) * sizeof(float));

	_draw_attention_scores = _draw_attention_scores_base + (b * (q_npos * kv_npos) * sizeof(float));
	_dattention_scores     = _dattention_scores_base     + (b * (q_npos * kv_npos) * sizeof(float));

	//_head_out = _head_out_base + (b * dim_stream *     q_npos * sizeof(float));
	_queries  = _queries_base  + (b * dim_qkv    *     q_npos * sizeof(float));
	_keys     = _keys_base     + (b * dim_qkv    * max_tokens * sizeof(float));
	_values   = _values_base   + (b * dim_qkv    * max_tokens * sizeof(float));

	_dhead_out = _dhead_out_base + (b * dim_stream *     q_npos * sizeof(float));
	_dqueries  = _dqueries_base  + (b * dim_qkv    *     q_npos * sizeof(float));
	_dkeys     = _dkeys_base     + (b * dim_qkv    * max_tokens * sizeof(float));
	_dvalues   = _dvalues_base   + (b * dim_qkv    * max_tokens * sizeof(float));




	if(attention_type == ATTENTION_FULL){
		
	   //#pragma omp parallel for private(i)	//NO! there would be a race condition over _dvalues in the output of multiply_vector_backward
	   for(size_t i=0; i<q_npos; i++){

		byte * _dw_value          = myalloc(dim_qkv * sizeof(float));


sum_vectors_backward(
	&_dhead_out[(i*dim_stream)*sizeof(float)], _dw_value,
	&_dhead_out[(i*dim_stream)*sizeof(float)],
	dim_qkv	
);


		for(size_t j = 0; j < kv_npos; j++){
			
			float w    = *(float *)&_attention_scores[((i*kv_npos)+j)*sizeof(float)];
			byte * dw  = &_dattention_scores[((i*kv_npos)+j)*sizeof(float)];

/*
			sum_vectors_backward(
				&_dhead_out[(i*dim_stream)*sizeof(float)], _dw_value,
				&_dhead_out[(i*dim_stream)*sizeof(float)],
				dim_qkv	
			);
*/
			multiply_vector_backward(
				&_dvalues[(j*dim_qkv)*sizeof(float)], dw,	
				&_values[(j*dim_qkv)*sizeof(float)], dim_qkv, w, _dw_value
			);

		}



		softmax_backward(
			&_draw_attention_scores[i*kv_npos*sizeof(float)],
			&_dattention_scores[i*kv_npos*sizeof(float)], &_attention_scores[i*kv_npos*sizeof(float)],
			1, kv_npos	
		);


		free(_dw_value);

	   }


	   //It would be like this, but having _draw_attention_scores as input and output is a mess...
	   //multiply_vector_backward(
	   //	_draw_attention_scores, NULL,
	   //	_raw_attention_scores, (q_npos * kv_npos), inv_div_k, _draw_attention_scores
	   //);

	   //...so, I just multiply the flowing gradient by the inv_div_k factor
	   multiply_vector(_draw_attention_scores, (q_npos * kv_npos), inv_div_k, _draw_attention_scores);

	   matmulf_nt_backward(
		_dkeys, _dqueries,
		_keys, _queries, dim_qkv,kv_npos, dim_qkv,q_npos, _draw_attention_scores,
		//B, T
		//1, 1
		1, q_npos
	   );


	}
	else
	if(attention_type == ATTENTION_CAUSAL){

	   //#pragma omp parallel for private(pos)	//NO! there would be a race condition over _dvalues in the output of multiply_vector_backward
	   for(size_t pos = q_startpos; pos < (q_startpos+q_npos); pos++){


	      if(pos>=kv_startpos){


		byte * _dw_value          = myalloc(dim_qkv * sizeof(float));


sum_vectors_backward(
	&_dhead_out[((pos-q_startpos)*dim_stream)*sizeof(float)], _dw_value,
	&_dhead_out[((pos-q_startpos)*dim_stream)*sizeof(float)],
	dim_qkv	
);


		//for(size_t j = 0; j < kv_npos; j++)					//NO!!!
		for(size_t j = 0; (j < kv_npos) && (j <= (pos-q_startpos)); j++)	//VERY, VERY IMPORTANT: we should NOT backpropagate gradients 
											//into positions not related (i.e.: AFTER) current query position,
											//even if we are managing this also in one more (different) way wight below
		{

			float w    = *(float *)&_attention_scores[(((pos-q_startpos)*kv_npos)+j)*sizeof(float)];
			byte * dw  = &_dattention_scores[(((pos-q_startpos)*kv_npos)+j)*sizeof(float)];

/*
			sum_vectors_backward(
				&_dhead_out[((pos-q_startpos)*dim_stream)*sizeof(float)], _dw_value,
				&_dhead_out[((pos-q_startpos)*dim_stream)*sizeof(float)],
				dim_qkv	
			);
*/

			multiply_vector_backward(
				&_dvalues[(j*dim_qkv)*sizeof(float)], dw,	
				&_values[(j*dim_qkv)*sizeof(float)], dim_qkv, w, _dw_value
			);

		}

#ifdef TRIP_DEBUG
wdebug(&_dattention_scores[T*pos*sizeof(float)], WTYPE_FLOAT32, 5, "_dattention_scores (datt_bth)",-1,0);
wdebug(_dvalues, WTYPE_FLOAT32, 5, "_dvalues",-1,1);
#endif



		softmax_backward(
			&_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)],
			&_dattention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], &_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)],
			1, (pos-kv_startpos+1)	
		);

#ifdef TRIP_DEBUG
wdebug(&_draw_attention_scores[T*pos*sizeof(float)], WTYPE_FLOAT32, 5, "_draw_attention_scores (datt_bth)",-1,1);
#endif

#ifdef TRIP_DEBUG
wdebug(&_dattention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], WTYPE_FLOAT32, kv_npos, "_dattention_scores",-1,0);
wdebug(&_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], WTYPE_FLOAT32, kv_npos, "_draw_attention_scores",-1,0);
#endif


		//It would be like this, but having _draw_attention_scores as input and output is a mess...
		//multiply_vector_backward(
		//	&_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], NULL,
		//	&_raw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], (pos-kv_startpos+1), inv_div_k, &_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)]
		//);

		//...so, I just multiply the flowing gradient by the inv_div_k factor
		multiply_vector(&_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], (pos-kv_startpos+1), inv_div_k, &_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)]);

#ifdef TRIP_DEBUG
wdebug(&_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)], WTYPE_FLOAT32, kv_npos, "_draw_attention_scores",-1,1);
#endif

		matmulf_nt_backward(
			&_dqueries[(pos-q_startpos)*dim_qkv*sizeof(float)], _dkeys,
			&_queries[(pos-q_startpos)*dim_qkv*sizeof(float)], _keys, dim_qkv,1, dim_qkv,(pos-kv_startpos+1), &_draw_attention_scores[(pos-q_startpos)*kv_npos*sizeof(float)],
			//B, T
			//1, 1
			1, (pos-kv_startpos+1)
		);

/*
if(pos==2){
wdebug(&_dqueries[(pos-q_startpos)*dim_qkv*sizeof(float)], WTYPE_FLOAT32, 5, "_dqueries",-1,0);
wdebug(&_dkeys[0*dim_qkv*sizeof(float)], WTYPE_FLOAT32, 5, "_dkeys",-1,1);
}
*/

		free(_dw_value);

  	      }

	   }

	}

    }	//loop over the batches



	wtype = bak_wtype;	//let's restore the general data type
	wsize = wtype_bytesize[wtype];	



	return 0;
}

// ============================================================
//  Positional embeddings
//
//  Transformers have no built-in sense of order — without
//  positional encoding, "the cat sat on the mat" and
//  "mat the on sat cat the" would look identical.
//  RoPE (Rotary Position Embedding) solves this by rotating
//  the query and key vectors as a function of their position,
//  so that the dot product Q·K naturally encodes relative
//  distance. TRiP also supports learned absolute embeddings
//  (encoder in PaliGemma) and the original sinusoidal
//  encoding (Vaswani).
// ============================================================





int apply_positional_embeddings(Model * model, byte * _out_vector, byte * _in_vector, size_t dim_vector, size_t pos_offset, size_t B, size_t T){

	float * out_vector = (float *)_out_vector;
	float * in_vector  = (float *)_in_vector;

	//LEARNED positional embeddings: the model checkpoint comes with pre-trained values of positional embeddings
	if(model->config.pose_cfg == POSE_LEARNED){
	
	  #pragma omp parallel for collapse(2)
	  for(size_t b = 0; b < B; b++){	
	     for(size_t t = 0; t < T; t++){	
		sum_vectors(
			&_in_vector[((b*T)+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32, 
			&model->w.learned_pose_w[(pos_offset + t)*dim_vector*wsize], wtype, 
			dim_vector, 
			&_out_vector[((b*T)+t)*dim_vector*sizeof(float)]
		);
	     }
	  }
	}
	else
	//VASWANI et al. postiional embeddings (original transformer)
	if(model->config.pose_cfg == POSE_ORIGINAL){

		//since we need to apply absolute sinusoidal encoding just before entering the first layer, and just to the token embedding,
		//we calculate just-in-time the encodings for this position. This will happen only once

	  #pragma omp parallel for collapse(2)
	  for(size_t b = 0; b < B; b++){	
	     for(size_t t = 0; t < T; t++){	
		for(size_t i=0; i<dim_vector; i+=2){
			float argval = ((float)(pos_offset + t))  /  powf(pose_theta, ((float)(i)) / ((float)dim_vector));
			out_vector[(((b*T)+t)*dim_vector) + i + 0]  =  in_vector[(((b*T)+t)*dim_vector) + i + 0]  +  sinf(argval);
			out_vector[(((b*T)+t)*dim_vector) + i + 1]  =  in_vector[(((b*T)+t)*dim_vector) + i + 1]  +  cosf(argval);
		}
	     }
	  }
	}
	else
	//Rotary Positional Embeddings (RoPE)
	if(model->config.pose_cfg == POSE_ROPE){

		if((action != ACTION_DECODE)  &&  (action != ACTION_CHAT)  &&  (action != ACTION_VISION)  &&  (action != ACTION_TRAIN)){

			mylog(LOG_ERROR,"RoPE positional embeddings are currently not supported by TRIP for the selected action (action code = 0x%02X)",action);
			exit(-1);
		}


		//if current size is not enough, reallocate memory for rope coefficients		
		if(rope_subk_timesteps < T){
			if(rope_subk != NULL){
				free(rope_subk);
				rope_subk = NULL;
				rope_subk_timesteps = 0;
			}
		}



		//let's allocate the memory for the RoPE coefficients, since this is the first time we use them
		if(rope_subk == NULL){
			

			if((action == ACTION_DECODE)  ||  (action == ACTION_CHAT)  ||  (action == ACTION_VISION)  ||  (action == ACTION_TRAIN)){
				//in the DECODE/CHAT modality, we do not need to store permanently all the sin/cos coefficients for EACH position,
				//as they are used only once: when processing the current position;
				//as matter of facts they will be used ONCE at each layer, but they are layer-invariant
	
				rope_subk = myalloc(((dim_vector/2) * T * (2*1)) * sizeof(float));	//note: we don't store the full rotation matrix, just the two coefficients sin and cos
				rope_subk_timesteps = T;
				rope_lastpos = -1;
			}
				
		}



		if((action == ACTION_DECODE)  ||  (action == ACTION_CHAT)  ||  (action == ACTION_VISION)  ||  (action == ACTION_TRAIN)){
			//if this is the first time we use RoPEs _at this position_ , let's calculate the coefficients;
			//we will use them at each layer, for each head, to calculate ("rotate") queries and keys for this position

			if(pos_offset != rope_lastpos){

	  		    #pragma omp parallel for
			    for(size_t t = 0; t < T; t++){

				size_t pos = pos_offset + t;

				for(size_t j=0; j<dim_vector; j+=2){
					float this_arg = ((float)pos) / (powf(pose_theta,((float)j)/((float)dim_vector)));
	
					rope_subk[(t*dim_vector) + j + 0] = cosf(this_arg);
					rope_subk[(t*dim_vector) + j + 1] = sinf(this_arg);
				}
			    }

			    rope_lastpos = pos_offset;
			}
		}



		if((action == ACTION_DECODE)  ||  (action == ACTION_CHAT)  ||  (action == ACTION_VISION)  ||  (action == ACTION_TRAIN)){

	  	  #pragma omp parallel for collapse(2)
		  for(size_t b = 0; b < B; b++){
		     for(size_t t = 0; t < T; t++){

			for(size_t j=0; j<dim_vector; j+=2){

				float rope_cos = rope_subk[(t*dim_vector) + j + 0];
				float rope_sin = rope_subk[(t*dim_vector) + j + 1];
				
				float v[2];

				v[0]  =  in_vector[(((b*T)+t)*dim_vector) + j + 0];
				v[1]  =  in_vector[(((b*T)+t)*dim_vector) + j + 1];
				out_vector[(((b*T)+t)*dim_vector) + j + 0]  =  (v[0]*rope_cos) - (v[1]*rope_sin);
				out_vector[(((b*T)+t)*dim_vector) + j + 1]  =  (v[0]*rope_sin) + (v[1]*rope_cos);
			}
		     }
		  }
		}

	}


	return 0;
}



int positional_embeddings_backward(
	byte * _din, byte * _dlearned,
	byte * _dout,  
	Model * model, size_t dim_vector, size_t pos_offset, 
	size_t B, size_t T
){

	float * din = (float *)_din;
	//float * dlearned = (float *)_dlearned;
	float * dout = (float *)_dout;

	//LEARNED positional embeddings: the model checkpoint comes with pre-trained values of positional embeddings
	if(model->config.pose_cfg == POSE_LEARNED){

	  //#pragma omp parallel for collapse(2)	//NO! there would be a race condition over dlearned
	  for(size_t b = 0; b < B; b++){	
	     for(size_t t = 0; t < T; t++){
		//backprop into learned embeddings
		sum_vectors(
			&_dlearned[(pos_offset+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32, 
			&_dout[((b*T)+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32,
			dim_vector, 
			&_dlearned[(pos_offset+t)*dim_vector*sizeof(float)] 
		);

		if(din!=dout){
		   //backprop into input 
		   sum_vectors(
			&_din[((b*T)+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32, 
			&_dout[((b*T)+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32,
			dim_vector, 
			&_din[((b*T)+t)*dim_vector*sizeof(float)] 
		   );
		}
	     }
	  }
	}
	else
	//VASWANI et al. positional embeddings (original transformer)
	if(model->config.pose_cfg == POSE_ORIGINAL){


	  #pragma omp parallel for collapse(2)
	  for(size_t b = 0; b < B; b++){	
	     for(size_t t = 0; t < T; t++){
		
		if(din!=dout){
		   //backprop into input 
		   sum_vectors(
			&_din[((b*T)+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32, 
			&_dout[((b*T)+t)*dim_vector*sizeof(float)], WTYPE_FLOAT32,
			dim_vector, 
			&_din[((b*T)+t)*dim_vector*sizeof(float)] 
		   );
     		}
	     }
	  }
	}
	else
	//Rotary Positional Embeddings (RoPE)
	if(model->config.pose_cfg == POSE_ROPE){

		//if current size is not enough, reallocate memory for rope coefficients		
		if(rope_subk_timesteps < T){
			if(rope_subk != NULL){
				free(rope_subk);
				rope_subk = NULL;
				rope_subk_timesteps = 0;
			}
		}


		//let's allocate the memory for the RoPE coefficients, since this is the first time we use them
		if(rope_subk == NULL){

			rope_subk = myalloc(((dim_vector/2) * T * (2*1)) * sizeof(float));	//note: we don't store the full rotation matrix, just the two coefficients sin and cos
			rope_subk_timesteps = T;
			rope_lastpos = -1;
		}



		//if this is the first time we use RoPEs _at this position_ , let's calculate the coefficients;
		//we will use them at each layer, for each head, to calculate ("rotate") queries and keys for this position

		if(pos_offset != rope_lastpos){

		    for(size_t t = 0; t < T; t++){

			size_t pos = pos_offset + t;

			for(size_t j=0; j<dim_vector; j+=2){
				float this_arg = ((float)pos) / (powf(pose_theta,((float)j)/((float)dim_vector)));

				rope_subk[(t*dim_vector) + j + 0] = cosf(this_arg);
				rope_subk[(t*dim_vector) + j + 1] = sinf(this_arg);
			}
		    }

		    rope_lastpos = pos_offset;
		}




	  	  #pragma omp parallel for collapse(2)
		  for(size_t b = 0; b < B; b++){
		     for(size_t t = 0; t < T; t++){

			for(size_t j=0; j<dim_vector; j+=2){

				float rope_cos = rope_subk[(t*dim_vector) + j + 0];
				float rope_sin = rope_subk[(t*dim_vector) + j + 1];

				/*				
				din[(((b*T)+t)*dim_vector) + j + 0] += (dout[(((b*T)+t)*dim_vector) + j + 0] * rope_cos);
				din[(((b*T)+t)*dim_vector) + j + 0] += (dout[(((b*T)+t)*dim_vector) + j + 1] * rope_sin);
				din[(((b*T)+t)*dim_vector) + j + 1] += (dout[(((b*T)+t)*dim_vector) + j + 0] * rope_sin) * -1.0;
				din[(((b*T)+t)*dim_vector) + j + 1] += (dout[(((b*T)+t)*dim_vector) + j + 1] * rope_cos);
				*/


				//I do this to avoid having to memorize query and key gradients in two steps during training;
				//this allows me to overwrite the gradient from output to input

				float din0 = 0.0;
				float din1 = 0.0;

				din0 += (dout[(((b*T)+t)*dim_vector) + j + 0] * rope_cos);
				din0 += (dout[(((b*T)+t)*dim_vector) + j + 1] * rope_sin);
				din1 += (dout[(((b*T)+t)*dim_vector) + j + 0] * rope_sin) * -1.0;
				din1 += (dout[(((b*T)+t)*dim_vector) + j + 1] * rope_cos);

				din[(((b*T)+t)*dim_vector) + j + 0]  =  din0;
				din[(((b*T)+t)*dim_vector) + j + 1]  =  din1;
			}
		     }
		  }

	}


	return 0;
}



int token_embeddings_backward(
	byte * _din, 
	byte * _dout, int ** intok, 
	Model * model, size_t B, size_t T
){


	size_t dim_stream = model->config.dim_stream;

/*
	float * din = (float *)_din;
	float * dout = (float *)_dout;
*/

	//#pragma omp parallel for collapse(2)	//NO! there COULD be a race condition over din
						//(even if it has b,t indexes, intok[b][t] is the actual index for din, and might be equal for different (b,t)s)
	for(size_t b = 0; b < B; b++){

		for(size_t t = 0; t < T; t++){

			//backprop into word embeddings
			sum_vectors(
				&_din[((size_t)intok[b][t]) * dim_stream * sizeof(float)], WTYPE_FLOAT32, 
				&_dout[((b*T)+t) * dim_stream * sizeof(float)],		   WTYPE_FLOAT32,
				dim_stream, 
				&_din[((size_t)intok[b][t]) * dim_stream * sizeof(float)] 
			);

		}
	}
	
	
	return 0;
}

// ============================================================
//  Softmax
//
//  Turns a raw vector of scores into a probability
//  distribution: exponentiate each element, then divide by
//  the sum. The max-subtraction trick prevents overflow.
//  Used in two places: attention scores and output logits.
// ============================================================




void softmax(byte * _a, size_t ax, size_t ay, byte * _b){
	size_t i,j;
	float max_val;
	float sum;



	float * a = (float *)_a;
	float * b = (float *)_b;

	for(j=0; j<ax; j++){
		//1) cerco il massimo fra tutte le componenti del vettore/matrice
 		max_val = *(a+(j*ay)+0);
		for(i=1; i<ay; i++){
        		if((*(a+(j*ay)+i)) > max_val) {
	        	    	max_val = (*(a+(j*ay)+i));
        		}
		}
    
		//2) compute each element of the output vector/matrix, not yet normalized
		//   NOTE: to reduce float overflow risk (high, since we compute expf(x)), we subtract the max;
		//         the final result (after dividing by "sum") is unchanged
		sum = 0.0;
		for(i=0; i<ay; i++){
			float val;
			val = expf((*(a+(j*ay)+i)) - max_val);
			*(b+(j*ay)+i) = val;
			sum += val;
		}

		for(i=0; i<ay; i++){
			*(b+(j*ay)+i) /= sum;
		}
    	}

}


void softmax_backward(
	byte * _din, 
	byte * _dout, byte * _out, 
	size_t ax, size_t ay
){


	float * dout = (float *)_dout;
	float *  out = (float *)_out;
	float *  din = (float *)_din;
    
  /* 
	//compute the backward pass for each row
	#pragma omp parallel for collapse(2)
	for(size_t j=0; j<ax; j++){
		for(size_t i=0; i<ay; i++){

			float grad = 0.0;

			local_derivative_ieqk  = out[(j*ay)+i] * (1.0 - out[(j*ay)+i]);
			//local_derivative_ineqk = out[(j*ay)+i] * (0.0 - out[(j*ay)+k]);

			for(size_t k=0; k<ay; k++){
				local_derivative  =  ((i==k)  ?  local_derivative_ieqk  :  (out[(j*ay)+i] * (0.0 - out[(j*ay)+k])));
				grad += (local_derivative * dout[(j*ay)+k]);
			}

			din[(j*ay)+i] += grad;
		}
	}
*/


	//compute the backward pass for each row
	#pragma omp parallel for
	for(size_t j=0; j<ax; j++){

		for(size_t i=0; i<ay; i++){

			for(size_t k=0; k<ay; k++){

				float indicator = ((i==k) ? 1.0 : 0.0);

				float local_derivative  =  (out[(j*ay)+i] * (indicator - out[(j*ay)+k]));
				din[(j*ay)+k] += (local_derivative * dout[(j*ay)+i]);
			}
		}
	}
}

// ============================================================
//  Cross-entropy loss
//
//  Measures how surprised the model is by the correct answer.
//  If the model assigns probability 1.0 to the right token,
//  loss is 0. If it assigns 0.0, loss is infinity.
// ============================================================



void crossentropy(float * calc_prob, float * target_prob, size_t size, float * loss){

	for(size_t i = 0; i < size; i++){

		loss[i] = -1.0 * ((target_prob[i]*logf(calc_prob[i])) + ((1.0-target_prob[i])*logf(1.0-calc_prob[i])));
	}

}

//This function backwards through softmax and crossentropy together, since this simplifies a lot the whole math
void crossentropy_softmax_backward(
	byte * _dlogits, 
	byte * _probs, int ** intok, 
	Model * model, int B, int T
){

	float * dlogits = (float *)_dlogits;
	float * probs   = (float *)_probs;


	size_t vs = (size_t)(model->config.vocab_size);	//vocabulary_size
	float dloss_mean = 1.0/((float)(B*T));		//this is the initial value of the chain rule, starting from the output of the loss function
							//it would be just "1.0" (dL/dL), but we want to initialize the backpropagation already averaging all the B*T losses together

	size_t nonpadded_vs = vs;

	if(checkpoint_type == CP_GPT2_AK){
                nonpadded_vs = 50257;	
	}


        #pragma omp parallel for collapse(2)
	for(size_t b = 0; b < B; b++){

		for(size_t t = 0; t < T; t++){


		        float * dlogits_bt = dlogits + (b*T*vs) + (t*vs);
	        	float * probs_bt = probs + (b*T*vs) + (t*vs);
		        size_t ix = intok[b][ t + 1 ];	//IMPORTANT: that's "t + 1" ! The target is the NEXT token, not the current one!

			if(intok[b][t  ] == toki.pad_id) continue;	//if current token is PAD, let's not backpropagate its contribution
			if(intok[b][t+1] == toki.pad_id) continue;	//if next token is PAD, let's not backpropagate the contribution of current token, because predicting PAD is nonsense

			//here we loop over all the vocabulary entries
			size_t i;
		        for(i = 0; i < nonpadded_vs; i++){
	        	        float p = probs_bt[i];
	                	float indicator = ((i==ix) ? 1.0 : 0.0);
		                dlogits_bt[i] += ((p - indicator) * dloss_mean);
	        	}


	        }
	}

}

// ============================================================
//  Vector operations  (multiply + sum)
//
//  The atomic arithmetic of the residual stream.
//  multiply_vector scales a vector by a scalar (e.g. used in
//  Gemma's embedding normalization). sum_vectors adds two
//  vectors element-wise, e.g. in the residual connection,
//  when the attention or FFN output is merged back into the
//  residual stream.
// ============================================================


void multiply_vector(byte * _va, size_t dim, float factor, byte * _vout){

	size_t i;

        float * va   = (float *)_va;
        float * vout = (float *)_vout;

	
        for(i=0; i < dim; i++){
                vout[i] = va[i] * factor;
        }

}

void multiply_vector_backward(
	byte * _dva, byte * _dfactor,
	byte * _va, size_t dim, float factor, byte * _dvout
){

	size_t i;

        float * va      = (float *)_va;
        float * dva     = (float *)_dva;
        float * dfactor = (float *)_dfactor;
        float * dvout   = (float *)_dvout;

	//forward pass:
        //vout[i] = va[i] * factor;

        for(i=0; i < dim; i++){
		dva[i] += dvout[i] * factor;
        }

	//if we need to backpropagate also into the multiplication factor (e.g.: inside the attention computation, when we scale each value vector by the attention score factor)
	if(dfactor!=NULL){
	        for(i=0; i < dim; i++){
			(*dfactor) += dvout[i] * va[i];
	        }
	}


}




void sum_vectors(byte * _va, int va_wtype, byte * _vb, int vb_wtype, size_t dim, byte * _vout){

	size_t i;

	float * vout = (float *)_vout;


	for(i=0; i < dim; i++){

		float va_i = (float)((va_wtype==WTYPE_FLOAT32) ? ((float *)_va)[i] : (va_wtype==WTYPE_BF16) ? ((__bf16 *)_va)[i] : (va_wtype==WTYPE_FLOAT16) ? ((_Float16 *)_va)[i] : 0.0);
		float vb_i = (float)((vb_wtype==WTYPE_FLOAT32) ? ((float *)_vb)[i] : (vb_wtype==WTYPE_BF16) ? ((__bf16 *)_vb)[i] : (vb_wtype==WTYPE_FLOAT16) ? ((_Float16 *)_vb)[i] : 0.0);

		vout[i] = va_i + vb_i;

	}

}

void sum_vectors_backward(
	byte * _da, byte * _db,
	byte * _dout,
	size_t dim
){

	float * da	= (float *)_da;
	float * db	= (float *)_db;
	float * dout	= (float *)_dout;

	if(da != dout){	//this check is because I may want to just propagate(reuse) the same gradient vector from output to input
		for(size_t i = 0; i < dim; i++){
			da[i] += dout[i];	
		}
	}

	if(db != dout){	//this check is because I may want to just propagate(reuse) the same gradient vector from output to input
		for(size_t i = 0; i < dim; i++){
			db[i] += dout[i];	
		}
	}
}

// ============================================================
//  Matrix multiplication
//
//  The single most expensive operation in the transformer.
//  Every projection (Q, K, V, O, FFN up, FFN down, logits)
//  is a matmul. TRiP implements both row x column (NN) and
//  row×row (NT, transposed) variants, with cache-line-aligned
//  tiling for performance. The interleaved variants reorder
//  the inner loop because "safetensors" format requires so
//  for Q and K projection matrices — and that was a real
//  headache until I discovered this fact.
// ============================================================






/*
#define SM (CACHE_LINESIZE / wsize)


//matmulf_nn: row × column product (non-transposed × non-transposed)


int matmulf_nn(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c){


	size_t k,ci,cj,cx,cy;

	if(ax!=by)	return -1;

	cx = bx;
	cy = ay;


//matmulf_nn as we know it so far: "a" matrix comes from checkpoint weights, "b" and "c" are always runstate vectors
      float * b = (float *)_b;
      float * c = (float *)_c;


	size_t axcj;


   if(wtype == WTYPE_FLOAT32){

      float * a = (float *)_a;


	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		axcj = ax*cj;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));

				val += (*(a+(axcj)+k)) * (*(b+(k*bx)+ci));
			}
			*(c+(cx*cj)+ci) = val;
		}
	}


   }
   else
   if(wtype == WTYPE_BF16){

      __bf16 * a = (__bf16 *)_a;


		
	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		axcj = ax*cj;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));
				float a_k = (float)(*(a+(axcj)+k));
				float b_k = (float)(*(b+(k*bx)+ci));
				val += a_k * b_k;
			}
			*(c+(cx*cj)+ci) = val;
		}
	}


   }
   else
   if(wtype == WTYPE_FLOAT16){

      _Float16 * a = (_Float16 *)_a;


		
	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		axcj = ax*cj;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));
				float a_k = (float)(*(a+(axcj)+k));
				float b_k = (float)(*(b+(k*bx)+ci));
				val += a_k * b_k;
			}
			*(c+(cx*cj)+ci) = val;
		}
	}


   }
 



	return 1;
}


int matmulf_nn_backward(
        byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, float * dc,
        float * da, float * db,
        size_t B, size_t T
){
        
        if(ax != by) return -1;
        
        size_t C = ax;   //number of columns in A, number of rows in B
        size_t OC = ay;  //number of rows in A and output C
        
	float * a = (float *)_a;
        float * b = (float *)_b;
        
        //backpropagation into input matrix "b": dB = A^T * dC
        //for each element in the batch and time sequence
        #pragma omp parallel for collapse(2)
        for(size_t bb = 0; bb < B; bb++){
                for(size_t tt = 0; tt < T; tt++){
                        //calculate pointers to the start of gradient matrices for this batch and timestep
                        //each dc matrix is ay×bx, each db matrix is by×bx
                        float * dc_bt = dc + (bb * T * ay * bx) + (tt * ay * bx);
                        float * db_bt = db + (bb * T * ax * bx) + (tt * ax * bx);
                        
                        for(size_t o = 0; o < ay; o++){
                        	float * dc_bto = dc_bt + (o * bx);
                                for(size_t i = 0; i < ax; i++){
                                	float a_oi = a[(o*ax) + i];
                                        for(size_t j = 0; j < bx; j++){
                                        	//for each column j in matrix B, accumulate gradient from all ay rows
                                                //db[i][j] += a[o][i] * dc[o][j]
                                                db_bt[i * bx + j] += a_oi * dc_bto[j];
                                        }
                                }
                        }
                }
        }
         
        size_t C = ax;   //number of columns in A, number of rows in B
        size_t OC = ay;  //number of rows in A and output C
        
        //backpropagation into input matrix "a": dA = dC * B^T
        //parallelize over rows of A
        #pragma omp parallel for
        for(size_t o = 0; o < ay; o++){
                //each row of da corresponds to a row in matrix A
                float * da_row = da + (o * ax);
                
                for(size_t bb = 0; bb < B; bb++){
                        for(size_t tt = 0; tt < T; tt++){
                                //get pointers to B and dC for this batch and timestep
                                float * b_bt = b + (bb * T * ax * bx) + (tt * ax * bx);
                                //dc_bto points to row o of dC for this batch and timestep
                                float * dc_bto = dc + (bb * T * ay * bx) + (tt * ay * bx) + (o * bx);
                                
                                for(size_t i = 0; i < ax; i++){
                                        //b_bti points to row i of B for this batch and timestep
                                        float * b_bti = b_bt + (i * bx);
                                        
                                        //for each element in B^T, accumulate into dA
                                        //da[o][i] += sum_j(dc[o][j] * b[i][j])
                                        for(size_t j = 0; j < bx; j++){
                                                da_row[i] += dc_bto[j] * b_bti[j];
                                        }
                                }
                        }
                }
        }
        
        return 1;
}
*/

/*
int matmulf_nn_fullf32(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c){


	size_t k,ci,cj,cx,cy;

	if(ax!=by)	return -1;

	cx = bx;
	cy = ay;


//matmulf_nn as we know it so far: "a" matrix comes from checkpoint weights, "b" and "c" are always runstate vectors
      float * b = (float *)_b;
      float * c = (float *)_c;


	size_t axcj;



      float * a = (float *)_a;

      if((cx==1)){

	#pragma omp parallel for private(cj)
	for(cj=0; cj<cy; cj++){
			
			float val = 0.0;
		
			axcj = ax*cj;
	
			for(k=0;k<ax;k++){	//row × column dot product
				val += a[axcj+k] * b[k];
			}
		
			c[cj] = val;

	}


      }
      else{

	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		axcj = ax*cj;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));

				val += (*(a+(axcj)+k)) * (*(b+(k*bx)+ci));
			}
			*(c+(cx*cj)+ci) = val;
		}
	}

      }



	return 1;
}
*/


/*
int matmulf_nn_interleaved(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c){


	size_t k,ci,cj,cx,cy;

	if(ax!=by)	return -1;

	cx = bx;
	cy = ay;

	size_t axcj;


//matmulf_nn as we know it so far: "a" matrix comes from checkpoint weights, "b" and "c" are always runstate vectors
      float * b = (float *)_b;
      float * c = (float *)_c;



   if(wtype == WTYPE_FLOAT32){

      float * a = (float *)_a;

      if((cx==1)){


	#pragma omp parallel for private(cj)
	for(cj=0; cj<cy; cj++){
			
			float val = 0.0;
		
			//axcj = ax*cj;
			axcj = (cj/2);
			if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
			axcj *= ax;
	
			for(k=0;k<ax;k++){	//row × column dot product
				val += a[axcj+k] * b[k];
			}
		
			c[cj] = val;

	}


      }
      else{

	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		//axcj = ax*cj;
		axcj = (cj/2);
		if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
		axcj *= ax;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));

				val += (*(a+(axcj)+k)) * (*(b+(k*bx)+ci));
			}
			*(c+(cx*cj)+ci) = val;
		}
	}

      }

   }
   else
   if(wtype == WTYPE_BF16){

      __bf16 * a = (__bf16 *)_a;


      if((cx==1)){


	#pragma omp parallel for private(cj)
	for(cj=0; cj<cy; cj++){
			
			float val = 0.0;


			//axcj = ax*cj;
			axcj = (cj/2);
			if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
			axcj *= ax;

			for(k=0;k<ax;k++){	//row × column dot product
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[k];
				val += a_k * b_k;
				//val += b[k] * (float)a[axcj+k];
			}
		
			c[cj] = val;

	}


      }
      else{

		
	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		//axcj = ax*cj;
		axcj = (cj/2);
		if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
		axcj *= ax;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));
				float a_k = (float)(*(a+(axcj)+k));
				float b_k = (float)(*(b+(k*bx)+ci));
				val += a_k * b_k;
			}
			*(c+(cx*cj)+ci) = val;
		}
	}

      }

   }
   else
   if(wtype == WTYPE_FLOAT16){

      _Float16 * a = (_Float16 *)_a;


      if((cx==1)){


	#pragma omp parallel for private(cj)
	for(cj=0; cj<cy; cj++){
			
			float val = 0.0;

			//axcj = ax*cj;
			axcj = (cj/2);
			if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
			axcj *= ax;

			for(k=0;k<ax;k++){	//row × column dot product
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[k];
				val += a_k * b_k;
			}
		
			c[cj] = val;

	}


      }
      else{

		
	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		//axcj = ax*cj;
		axcj = (cj/2);
		if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
		axcj *= ax;

		for(ci=0;ci<cx;ci++){
			//c[cj][ci] = 0;		//formally it's like this
			//*(c+(cx*cj)+ci) = 0.0;	//but it would be like this
			float val = 0.0;		//but in practice it is better to accumulate locally, and then transfer the final result to the RAM
			for(k=0;k<ax;k++){	//row × column dot product
				//c[cj][ci] += (a[cj][k]*b[k][ci]);
				//*(c+(cx*cj)+ci) += (*(a+(ax*cj)+k)) * (*(b+(k*bx)+ci));
				float a_k = (float)(*(a+(axcj)+k));
				float b_k = (float)(*(b+(k*bx)+ci));
				val += a_k * b_k;
			}
			*(c+(cx*cj)+ci) = val;
		}
	}

      }

   }
 



	return 1;
}
*/



//matmulf_nt: row x row product (equivalent to row x column between non-transposed and transposed)

int matmulf_nt(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c){

	size_t cx,cy;

	if(ax!=bx)	return -1;

	cx = by;
	cy = ay;


	float * b = (float *)_b;
	float * c = (float *)_c;


 if(wtype == WTYPE_FLOAT32){

 	float * a = (float *)_a;


	#pragma omp parallel for
	for(size_t cj=0;cj<cy;cj++){

		size_t axcj;
		size_t bxci;


		axcj = ax*cj;

		for(size_t ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(size_t k=0;k<ax;k++){	//row × row dot product (transposed column)
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[bxci+k];
				val += a_k * b_k;
			}

			c[(cy*ci)+cj] = val;
		}
	}
}
else
if(wtype == WTYPE_BF16){

	__bf16 * a = (__bf16 *)_a;

	#pragma omp parallel for
	for(size_t cj=0;cj<cy;cj++){

		size_t axcj;
		size_t bxci;


		axcj = ax*cj;

		for(size_t ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(size_t k=0;k<ax;k++){	//row x row dot product (transposed column)
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[bxci+k];
				val += a_k * b_k;
			}

			c[(cy*ci)+cj] = val;
		}
	}

}
else
if(wtype == WTYPE_FLOAT16){

	_Float16 * a = (_Float16 *)_a;

	#pragma omp parallel for
	for(size_t cj=0;cj<cy;cj++){

		size_t axcj;
		size_t bxci;


		axcj = ax*cj;

		for(size_t ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(size_t k=0;k<ax;k++){	//row x row dot product (transposed column)
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[bxci+k];
				val += a_k * b_k;
			}

			c[(cy*ci)+cj] = val;
		}
	}


}




	return 1;

}



int matmulf_nt_backward(
	byte * _da, byte * _db, 
	byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _dc,
	size_t B, size_t T
){

	float * da = (float *)_da;
	float * db = (float *)_db;
	float * dc = (float *)_dc;


	size_t C  = ax;	// C =  input channels
	size_t OC = ay;	//OC = output channels

	float * b = (float *)_b;


	//backpropagation into the input vector "b"
	#pragma omp parallel for collapse(2)
	for(size_t bb = 0; bb < B; bb++){

		for(size_t tt = 0; tt < T; tt++){

			float * dc_bt  =  dc + (bb * T * OC) + (tt * OC);
			float * db_bt  =  db + (bb * T * C ) + (tt * C );
	
	
			for(size_t o = 0; o < OC; o++){

				float dc_bto  =  dc_bt[o];
		
				if(wtype == WTYPE_FLOAT32){

					float * a = (float *)_a;					
					float a_oi;

					for(size_t i = 0; i < C; i++){
						a_oi      = (float)a[(o*C)+i];
						db_bt[i] += (a_oi * dc_bto);
					}
				}
				else
				if(wtype == WTYPE_BF16){
		
					__bf16 * a = (__bf16 *)_a;					
					float a_oi;

					for(size_t i = 0; i < C; i++){
						a_oi      = (float)a[(o*C)+i];
						db_bt[i] += (a_oi * dc_bto);
					}
				}
				else
				if(wtype == WTYPE_FLOAT16){		

					_Float16 * a = (_Float16 *)_a;					
					float a_oi;	

					for(size_t i = 0; i < C; i++){
						a_oi      = (float)a[(o*C)+i];
						db_bt[i] += (a_oi * dc_bto);
					}
				}

			}
		
		}
	}




	//backpropagation into input weights matrix "a"
	#pragma omp parallel for
	for(size_t o = 0; o < OC; o++){

                for(size_t bb = 0; bb < B; bb++){

                        for(size_t tt = 0; tt < T; tt++){

                                float * b_bt    =  b  + (bb * T * C ) + (tt * C );
                                float * da_row  =  da + (o * C);
                                float   dc_bto  =  dc[(bb * T * OC) + (tt * OC) + o];


                                for(size_t i = 0; i < C; i++){
                                        da_row[i] += (b_bt[i] * dc_bto);
                                }
                        }
                }
        }
}



int matmulf_nt_interleaved(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c){

	size_t cx,cy;

	if(ax!=bx)	return -1;

	cx = by;
	cy = ay;

	float * b = (float *)_b;
	float * c = (float *)_c;


 if(wtype == WTYPE_FLOAT32){

 	float * a = (float *)_a;

	#pragma omp parallel for
	for(size_t cj=0;cj<cy;cj++){

		size_t axcj;
		size_t bxci;


		//axcj = ax*cj;
		axcj = (cj/2);
		if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
		axcj *= ax;


		for(size_t ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(size_t k=0;k<ax;k++){	//row x row dot product (transposed column)
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[bxci+k];
				val += a_k * b_k;
			}

			c[(cy*ci)+cj] = val;
		}
	}

}
else
if(wtype == WTYPE_BF16){

	__bf16 * a = (__bf16 *)_a;

	#pragma omp parallel for
	for(size_t cj=0;cj<cy;cj++){

		size_t axcj;
		size_t bxci;


		//axcj = ax*cj;
		axcj = (cj/2);
		if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
		axcj *= ax;


		for(size_t ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(size_t k=0;k<ax;k++){	//row x row dot product (transposed column)
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[bxci+k];
				val += a_k * b_k;
			}

			c[(cy*ci)+cj] = val;
		}
	}

}
else
if(wtype == WTYPE_FLOAT16){

	_Float16 * a = (_Float16 *)_a;

	#pragma omp parallel for
	for(size_t cj=0;cj<cy;cj++){

		size_t axcj;
		size_t bxci;


		//axcj = ax*cj;
		axcj = (cj/2);
		if((cj%2)==1)	axcj+=(cy/2);	//let's skip to the odd-rows submatrix, which in safetensors/hf format is in the second half of the matrix
		axcj *= ax;


		for(size_t ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(size_t k=0;k<ax;k++){	//row x row dot product (transposed column)
				float a_k = (float)a[axcj+k];
				float b_k = (float)b[bxci+k];
				val += a_k * b_k;
			}

			c[(cy*ci)+cj] = val;
		}
	}


}




	return 1;

}


int matmulf_nt_interleaved_backward(
	byte * _da, byte * _db, 
	byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _dc,
	size_t B, size_t T
){

	float * da = (float *)_da;
	float * db = (float *)_db;
	float * dc = (float *)_dc;


	size_t C  = ax;	// C =  input channels
	size_t OC = ay;	//OC = output channels


	float * b = (float *)_b;


	//backpropagation into the input vector "b"
	#pragma omp parallel for collapse(2)
	for(size_t bb = 0; bb < B; bb++){

		for(size_t tt = 0; tt < T; tt++){

			float * dc_bt  =  dc + (bb * T * OC) + (tt * OC);
			float * db_bt  =  db + (bb * T * C ) + (tt * C );
	
	
			for(size_t o = 0; o < OC; o++){

				float dc_bto  =  dc_bt[o];

				size_t o_interleaved;
				o_interleaved = (o/2);
				if((o%2)==1)    o_interleaved += (OC/2);	//let's skip to the odd-rows submatrix, 
										//which in safetensors/hf format is in the second half of the matrix
		
				if(wtype == WTYPE_FLOAT32){

					float * a = (float *)_a;					
					float a_oi;

					for(size_t i = 0; i < C; i++){

						a_oi      = (float)a[(o_interleaved*C)+i];
						db_bt[i] += (a_oi * dc_bto);
					}
				}
				else
				if(wtype == WTYPE_BF16){
		
					__bf16 * a = (__bf16 *)_a;					
					float a_oi;

					for(size_t i = 0; i < C; i++){
	
						a_oi      = (float)a[(o_interleaved*C)+i];
						db_bt[i] += (a_oi * dc_bto);
					}
				}
				else
				if(wtype == WTYPE_FLOAT16){		

					_Float16 * a = (_Float16 *)_a;					
					float a_oi;	

					for(size_t i = 0; i < C; i++){

						a_oi      = (float)a[(o_interleaved*C)+i];
						db_bt[i] += (a_oi * dc_bto);
					}
				}

			}
		
		}
	}


        //backpropagation into input weights matrix "a"
        #pragma omp parallel for
	for(size_t o = 0; o < OC; o++){

                for(size_t bb = 0; bb < B; bb++){

                        for(size_t tt = 0; tt < T; tt++){

                                size_t o_interleaved;
                                o_interleaved = (o/2);
                                if((o%2)==1)    o_interleaved += (OC/2);        //let's skip to the odd-rows submatrix,
                       								//which in safetensors/hf format is in the second half of the matrix
				float * b_bt    =  b  + (bb * T * C ) + (tt * C );
                                float * da_row  =  da + (o_interleaved * C);
                                float   dc_bto  =  dc[(bb * T * OC) + (tt * OC) + o];


                                for(size_t i = 0; i < C; i++){
                                        da_row[i] += (b_bt[i] * dc_bto);
                                }
                        }
                }
        }

}


/*
int matmulf_nt_fullf32(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c){

	size_t k,ci,cj,cx,cy;

	if(ax!=bx)	return -1;

	cx = by;
	cy = ay;


	size_t axcj;
	size_t bxci;

	float * a = (float *)_a;
	float * b = (float *)_b;
	float * c = (float *)_c;


	#pragma omp parallel for private(cj)
	for(cj=0;cj<cy;cj++){

		axcj = ax*cj;

		for(ci=0;ci<cx;ci++){
		
			bxci = bx*ci;

			float val = 0.0;

			for(k=0;k<ax;k++){	//row × row dot product (transposed column)
				val += a[axcj+k] * b[bxci+k];
			}

			//c[(cx*cj)+ci] = val;
			c[(cy*ci)+cj] = val;
		}
	}


	return 1;

}

*/



/*
int test_matmulf_nn(){

	int testval = 0;

   {
	float a[5][4] = {
		{ 1,2,3,4 },
		{ 5,6,7,8 },
		{ 9,10,11,12 },
		{ 13,14,15,16 },
		{ 17,18,19,20 }
	};
	float b[4][3] = {
		{ 21,22,23 },
		{ 24,25,26 },
		{ 27,28,29 },
		{ 30,31,32 }
	};
	float c[5][3];
	float c_true[5][3] = {
		{ 270,280,290 },
		{ 678,704,730 },
		{ 1086,1128,1170 },
		{ 1494,1552,1610 },
		{ 1902,1976,2050 }
	};


	wtype = WTYPE_FLOAT32;
	wsize = wtype_bytesize[wtype]; 

	mylog(LOG_VERBOSE_DEBUG,"test_matmulf_nn() using %s",wtype_text[wtype]);

	matmulf_nn((byte *)a,(byte *)b,4,5,3,4,(byte *)c);
	

	for(int i=0;i<5;i++){
		for(int j=0;j<3;j++){
			mylog(LOG_VERBOSE_DEBUG,"c%d%d = %f \t c_true%d%d = %f \t %s",i+1,j+1,c[i][j],i+1,j+1,c_true[i][j],(c[i][j]==c_true[i][j])?"ok":"KO!");
			if(c[i][j]!=c_true[i][j])	testval--;
		}
	}

   }


   {
	__bf16 a[5][4] = {
		{ 1,2,3,4 },
		{ 5,6,7,8 },
		{ 9,10,11,12 },
		{ 13,14,15,16 },
		{ 17,18,19,20 }
	};
	float b[4][3] = {
		{ 21,22,23 },
		{ 24,25,26 },
		{ 27,28,29 },
		{ 30,31,32 }
	};
	float c[5][3];
	float c_true[5][3] = {
		{ 270,280,290 },
		{ 678,704,730 },
		{ 1086,1128,1170 },
		{ 1494,1552,1610 },
		{ 1902,1976,2050 }
	};



	wtype = WTYPE_BF16;
	wsize = wtype_bytesize[wtype]; 

	mylog(LOG_VERBOSE_DEBUG,"test_matmulf_nn() using %s",wtype_text[wtype]);

	matmulf_nn((byte *)a,(byte *)b,4,5,3,4,(byte *)c);
	

	for(int i=0;i<5;i++){
		for(int j=0;j<3;j++){
			mylog(LOG_VERBOSE_DEBUG,"c%d%d = %f \t c_true%d%d = %f \t %s",i+1,j+1,(float)c[i][j],i+1,j+1,(float)c_true[i][j],(c[i][j]==c_true[i][j])?"ok":"KO!");
			if(c[i][j]!=c_true[i][j])	testval--;
		}
	}

   }


   {
	_Float16 a[5][4] = {
		{ 1,2,3,4 },
		{ 5,6,7,8 },
		{ 9,10,11,12 },
		{ 13,14,15,16 },
		{ 17,18,19,20 }
	};
	float b[4][3] = {
		{ 21,22,23 },
		{ 24,25,26 },
		{ 27,28,29 },
		{ 30,31,32 }
	};
	float c[5][3];
	float c_true[5][3] = {
		{ 270,280,290 },
		{ 678,704,730 },
		{ 1086,1128,1170 },
		{ 1494,1552,1610 },
		{ 1902,1976,2050 }
	};



	wtype = WTYPE_FLOAT16;
	wsize = wtype_bytesize[wtype]; 

	mylog(LOG_VERBOSE_DEBUG,"test_matmulf_nn() using %s",wtype_text[wtype]);

	matmulf_nn((byte *)a,(byte *)b,4,5,3,4,(byte *)c);
	

	for(int i=0;i<5;i++){
		for(int j=0;j<3;j++){
			mylog(LOG_VERBOSE_DEBUG,"c%d%d = %f \t c_true%d%d = %f \t %s",i+1,j+1,(float)c[i][j],i+1,j+1,(float)c_true[i][j],(c[i][j]==c_true[i][j])?"ok":"KO!");
			if(c[i][j]!=c_true[i][j])	testval--;
		}
	}

   }


	return testval;
}

int test_matmulf_nt(){

	int testval = 0;

   {
	float a[5][4] = {
		{ 1,2,3,4 },
		{ 5,6,7,8 },
		{ 9,10,11,12 },
		{ 13,14,15,16 },
		{ 17,18,19,20 }
	};
	//float b[4][3] = {
	//	{ 21,22,23 },
	//	{ 24,25,26 },
	//	{ 27,28,29 },
	//	{ 30,31,32 }
	//};
	float b[3][4] = {
		{ 21,24,27,30 },
		{ 22,25,28,31 },
		{ 23,26,29,32 }
	};
	float c[5][3];
	float c_true[5][3] = {
		{ 270,280,290 },
		{ 678,704,730 },
		{ 1086,1128,1170 },
		{ 1494,1552,1610 },
		{ 1902,1976,2050 }
	};


	wtype = WTYPE_FLOAT32;
	wsize = wtype_bytesize[wtype]; 


	mylog(LOG_VERBOSE_DEBUG,"test_matmulf_nt()) using %s",wtype_text[wtype]);

	matmulf_nt((byte *)a,(byte *)b,4,5,4,3,(byte *)c);
	
	int testval = 0;

	for(int i=0;i<5;i++){
		for(int j=0;j<3;j++){
			mylog(LOG_VERBOSE_DEBUG,"c%d%d = %f \t c_true%d%d = %f \t %s",i+1,j+1,c[i][j],  i+1,j+1,c_true[i][j],  (c[i][j]==c_true[i][j])?"ok":"KO!");
			if(c[i][j]!=c_true[i][j])	testval--;
		}
	}

   }



	return testval;
}
*/



// ============================================================
//  Layernorm 
//
//  Centers and rescales the residual stream, so that each position has
//  mean 0 and variance 1, then applies learned scale and bias.
//  Keeps gradients healthy across deep layer stacks.
//  Used by GPT-2 and similar architectures.
// ============================================================

int layernorm(byte * _out, byte * _in, byte * _output_weights, byte * _output_bias, size_t dim, size_t B, size_t T, byte * _mean, byte * _rstd){


	if(dim==0)	 return -1;


	#pragma omp parallel for collapse(2)
	for(size_t b = 0; b < B; b++){
		for(size_t t = 0; t < T; t++){

		   float * out            = (float *)( &_out[((b*T)+t)*dim*sizeof(float)]);
		   float * in             = (float *)(  &_in[((b*T)+t)*dim*sizeof(float)]);

		   float mean = 0.0;
		   float variance = 0.0;
		   float inv_stdev;
		
		   if(wtype == WTYPE_FLOAT32){
		
			float * output_weights = (float *)_output_weights;
			float * output_bias    = (float *)_output_bias;
		
		
			for(size_t i=0;i<dim;i++){
				mean += in[i];
			}
			mean /= (float)dim;
		
			for(size_t i=0;i<dim;i++){
				float in_nobias;
				in_nobias = in[i]-mean;
				variance += (in_nobias*in_nobias);
			}
			variance /= (float)dim;
		
		
			variance += norm_eps;
		
		
			//reciprocal of the standard deviation
			inv_stdev = 1.0/sqrtf(variance);
		
		
			for(size_t i=0;i<dim;i++){
				float temp;
				temp = (in[i]-mean) * inv_stdev;
				temp = output_bias[i]  +  (output_weights[i] * temp);
				out[i] = temp;	//final write to RAM
			}
		   }
		   else
		   if(wtype == WTYPE_BF16){
		
			__bf16 * output_weights = (__bf16 *)_output_weights;
			__bf16 * output_bias    = (__bf16 *)_output_bias;
		
		
			for(size_t i=0;i<dim;i++){
				mean += (float)in[i];
			}
			mean /= (float)dim;
		
			for(size_t i=0;i<dim;i++){
				float in_nobias;
				in_nobias = ((float)in[i]) - mean;
				variance += (in_nobias*in_nobias);
			}
			variance /= (float)dim;
		
		
			variance += norm_eps;
		
		
			//reciprocal of the standard deviation
			inv_stdev = 1.0/sqrtf(variance);
		
			for(size_t i=0;i<dim;i++){
				float temp;
				float ow_i = (float)output_weights[i];
				float ob_i = (float)output_bias[i];
				temp = (((float)in[i]) - mean) * inv_stdev;
				temp = ob_i  +  (ow_i * temp);
				out[i] = temp;	//final write to RAM
			}
		   }
		   else
		   if(wtype == WTYPE_FLOAT16){
		
			_Float16 * output_weights = (_Float16 *)_output_weights;
			_Float16 * output_bias    = (_Float16 *)_output_bias;
		
		
			for(size_t i=0;i<dim;i++){
				mean += (float)in[i];
			}
			mean /= (float)dim;
		
			for(size_t i=0;i<dim;i++){
				float in_nobias;
				in_nobias = ((float)in[i]) - mean;
				variance += (in_nobias*in_nobias);
			}
			variance /= (float)dim;
		
			
			variance += norm_eps;
		
		
			//reciprocal of the standard deviation
			inv_stdev = 1.0/sqrtf(variance);
		
			for(size_t i=0;i<dim;i++){
				float temp;
				float ow_i = (float)output_weights[i];
				float ob_i = (float)output_bias[i];
				temp = (((float)in[i]) - mean) * inv_stdev;
				temp = ob_i  +  (ow_i * temp);
				out[i] = temp;	//final write to RAM
			}
		   }

		   if(action == ACTION_TRAIN){
			float * out_mean;
			float * out_rstd;

			out_mean       = (float *)(&_mean[((b*T)+t)*    sizeof(float)]);
			out_rstd       = (float *)(&_rstd[((b*T)+t)*    sizeof(float)]);

			*out_mean = mean;
			*out_rstd = inv_stdev;
		   }
		}
	}	 
		
		
		
	return 0;
}


int layernorm_backward(
	byte * _din, byte * _dweight, byte * _dbias,
	byte * _dout, byte * _in, byte * _output_weights, byte * _mean, byte * _rstd,
	size_t B, size_t T, size_t dim
){

	float * din = (float *)_din;
	float * dweight = (float *)_dweight;
	float * dbias = (float *)_dbias;
	float * dout = (float *)_dout;
	float * mean = (float *)_mean;
	float * rstd = (float *)_rstd;

	float * dweight_bt = (float *)myalloc(dim * sizeof(float) * B * T);
	float * dbias_bt   = (float *)myalloc(dim * sizeof(float) * B * T);


	size_t C = dim;	//number of channels (same in input and output)

	float * in = (float *)_in;

	#pragma omp parallel for collapse(2)
	for(size_t b = 0; b < B; b++){

		for(size_t t = 0; t < T; t++){

			//offsets
			float * dout_bt = dout + (b * T * C) + (t * C);
			float * in_bt   = in   + (b * T * C) + (t * C);
			float * din_bt  = din  + (b * T * C) + (t * C);

			float mean_bt = mean[(b * T) + t];
			float rstd_bt = rstd[(b * T) + t];

			//first: two reduce operations
			float dnorm_mean = 0.0f;
			float dnorm_norm_mean = 0.0f;

			for(size_t i = 0; i < C; i++){

				float ow_i;

				if(wtype == WTYPE_FLOAT32){
					float * ow = (float *)_output_weights;
					ow_i = (float)ow[i];
				}
				else
				if(wtype == WTYPE_BF16){
					__bf16 * ow = (__bf16 *)_output_weights;
					ow_i = (float)ow[i];
				}
				else
				if(wtype == WTYPE_FLOAT16){
					_Float16 * ow = (_Float16 *)_output_weights;
					ow_i = (float)ow[i];
				}
			

				float norm_bti = (in_bt[i] - mean_bt) * rstd_bt;
		                float dnorm_i  = ow_i * dout_bt[i];

                		dnorm_mean += dnorm_i;
		                dnorm_norm_mean += dnorm_i * norm_bti;
            		}

			dnorm_mean = dnorm_mean / C;
			dnorm_norm_mean = dnorm_norm_mean / C;

			//now: iterate again, and accumulate all the gradients
			for(size_t i = 0; i < C; i++){

				float ow_i;

				if(wtype == WTYPE_FLOAT32){
					float * ow = (float *)_output_weights;
					ow_i = (float)ow[i];
				}
				else
				if(wtype == WTYPE_BF16){
					__bf16 * ow = (__bf16 *)_output_weights;
					ow_i = (float)ow[i];
				}
				else
				if(wtype == WTYPE_FLOAT16){
					_Float16 * ow = (_Float16 *)_output_weights;
					ow_i = (float)ow[i];
				}
			

				float norm_bti = (in_bt[i] - mean_bt) * rstd_bt;
				float dnorm_i  = ow_i * dout_bt[i];

				//gradient contribution to bias
				dbias_bt[(((b*T)+t)*C)+i] += dout_bt[i];
				//gradient contribution to weight
				dweight_bt[(((b*T)+t)*C)+i] += norm_bti * dout_bt[i];
				//gradient contribution to input
				float dval = 0.0f;
				dval += dnorm_i; // term 1
				dval -= dnorm_mean; // term 2
				dval -= norm_bti * dnorm_norm_mean; // term 3
				dval *= rstd_bt; // final scale
				din_bt[i] += dval;
			}
		}
	}



	//reduction to dweight and dbias
        for(size_t i = 0; i < C; i++){

		float val_w = 0.0;
		float val_b = 0.0;

        	for(size_t b = 0; b < B; b++){
                	for(size_t t = 0; t < T; t++){
				val_w += dweight_bt[(((b*T) + t) * C) + i];
				val_b +=   dbias_bt[(((b*T) + t) * C) + i];
			}
		}

		dweight[i] += val_w;
		dbias[i]   += val_b;
	}



	free(dweight_bt);
	free(dbias_bt);

	return 0;
}


// ============================================================
//  RMSnorm
//
//  A simpler cousin of LayerNorm: skip the mean subtraction,
//  just divide by the root-mean-square, then scale.
//  Cheaper to compute; that's what Llama and Gemma actually use.
// ============================================================

int rmsnorm(Model * model, byte * _out, byte * _in, byte * _output_weights, size_t dim, size_t B, size_t T, byte * _rrms){


	if(dim==0)	 return -1;


	#pragma omp parallel for collapse(2)
	for(size_t b = 0; b < B; b++){
		for(size_t t = 0; t < T; t++){

		   size_t i;

		   float * out            = (float *)( &_out[((b*T)+t)*dim*sizeof(float)]);
		   float * in             = (float *)(  &_in[((b*T)+t)*dim*sizeof(float)]);


		   float rms = 0.0;

	
		   if(wtype == WTYPE_FLOAT32){
		
			float * output_weights = (float *)_output_weights;
		
			for(i=0;i<dim;i++){
				rms += (in[i]*in[i]);
			}
		
			rms /= (float)dim;
			rms += norm_eps;
			rms  = sqrtf(rms);
		
		
			rms = 1.0/rms;
		
		
			if( (model->config.architectures == ARCH_GEMMA_CAUSAL)
			    ||
			    (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)
			){
				for(i=0;i<dim;i++){
					out[i] = (1.0 + output_weights[i]) * rms * in[i];
				}
			}
			else{
				for(i=0;i<dim;i++){
					out[i] = output_weights[i] * rms * in[i];
				}
			}
		   }
		   else
		   if(wtype == WTYPE_BF16){
		
			__bf16 * output_weights = (__bf16 *)_output_weights;
		
			for(i=0;i<dim;i++){
				rms += (in[i]*in[i]);
			}
		
			rms /= (float)dim;
			rms += norm_eps;
			rms  = sqrtf(rms);
		
		
			rms = 1.0/rms;
		
		
			if( (model->config.architectures == ARCH_GEMMA_CAUSAL)
			    ||
			    (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)
			){
				for(i=0;i<dim;i++){
					float ow_i = (float)output_weights[i];
					out[i] = (1.0 + ow_i) * rms * in[i];
				}
			}
			else{
				for(i=0;i<dim;i++){
					float ow_i = (float)output_weights[i];
					out[i] = ow_i * rms * in[i];
				}
			}
		   }
		   else
		   if(wtype == WTYPE_FLOAT16){
		
			_Float16 * output_weights = (_Float16 *)_output_weights;
		
			for(i=0;i<dim;i++){
				rms += (in[i]*in[i]);
			}
		
			rms /= (float)dim;
			rms += norm_eps;
			rms  = sqrtf(rms);
		
		
			rms = 1.0/rms;
		
		
			if( (model->config.architectures == ARCH_GEMMA_CAUSAL)
			    ||
			    (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)
			){
				for(i=0;i<dim;i++){
					float ow_i = (float)output_weights[i];
					out[i] = (1.0 + ow_i) * rms * in[i];
				}
			}
			else{
				for(i=0;i<dim;i++){
					float ow_i = (float)output_weights[i];
					out[i] = ow_i * rms * in[i];
				}
			}
		
		   }

		   if(action == ACTION_TRAIN){
		   	float * out_rrms;
		   	out_rrms  = (float *)(&_rrms[((b*T)+t)*    sizeof(float)]);
			*out_rrms = rms; //"rms" is already 1/rms here
		   }
		}
	}	 
		
		
		
	return 0;
}


int rmsnorm_backward(
        byte * _din, byte * _dweight,
        byte * _dout, byte * _in, byte * _output_weights, byte * _rrms,
        Model * model, size_t B, size_t T, size_t dim
){

	float * din = (float *)_din;
	float * dweight = (float *)_dweight;
	float * dout = (float *)_dout;
	float * rrms = (float *)_rrms;


	float * dweight_bt = (float *)myalloc(dim * sizeof(float) * B * T);


        size_t C = dim; //number of channels (same in input and output)
        float * in = (float *)_in;
    
	#pragma omp parallel for collapse(2)    
        for(size_t b = 0; b < B; b++){

                for(size_t t = 0; t < T; t++){

                        //offsets
                        float * dout_bt = dout + (b * T * C) + (t * C);
                        float * in_bt   = in   + (b * T * C) + (t * C);
                        float * din_bt  = din  + (b * T * C) + (t * C);
                        float rrms_bt = rrms[(b * T) + t];
                        
                        //first: compute the sum for RMS gradient term
                        float sum_wxdout = 0.0;
                        for(size_t i = 0; i < C; i++){
                                float ow_i;
                                if(wtype == WTYPE_FLOAT32){
                                        float * ow = (float *)_output_weights;
                                        ow_i = ow[i];
                                }
                                else
                                if(wtype == WTYPE_BF16){
                                        __bf16 * ow = (__bf16 *)_output_weights;
                                        ow_i = (float)ow[i];
                                }
                                else
                                if(wtype == WTYPE_FLOAT16){
                                        _Float16 * ow = (_Float16 *)_output_weights;
                                        ow_i = (float)ow[i];
                                }
                                
                                //handle Gemma/PaliGemma architectures
                                bool is_special_arch = (model->config.architectures == ARCH_GEMMA_CAUSAL || 
                                                      model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL);
                                float scale_factor = is_special_arch ? (1.0 + ow_i) : ow_i;
                                
                                sum_wxdout += scale_factor * in_bt[i] * dout_bt[i];
                        }
                        
                        //now: iterate again, and compute all gradients
                        for(size_t i = 0; i < C; i++){
                                float ow_i;
                                if(wtype == WTYPE_FLOAT32){
                                        float * ow = (float *)_output_weights;
                                        ow_i = ow[i];
                                }
                                else
                                if(wtype == WTYPE_BF16){
                                        __bf16 * ow = (__bf16 *)_output_weights;
                                        ow_i = (float)ow[i];
                                }
                                else
                                if(wtype == WTYPE_FLOAT16){
                                        _Float16 * ow = (_Float16 *)_output_weights;
                                        ow_i = (float)ow[i];
                                }
                                
                                //handle Gemma/PaliGemma architectures
                                bool is_special_arch = (model->config.architectures == ARCH_GEMMA_CAUSAL || 
                                                      model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL);
                                float scale_factor = is_special_arch ? (1.0f + ow_i) : ow_i;
                                
                                //gradient contribution to weight: dout * x / rms
                                dweight_bt[(((b*T) + t) * C) + i] += in_bt[i] * rrms_bt * dout_bt[i];
                                
                                //gradient contribution to input
                                float dval = scale_factor * rrms_bt * dout_bt[i]; //direct term
                                float rms_grad = -(in_bt[i] * rrms_bt * rrms_bt * rrms_bt / C) * sum_wxdout; //RMS term
                                dval += rms_grad;
                                din_bt[i] += dval;
                        }
                }
        }


	//reduction to dweight
        for(size_t i = 0; i < C; i++){
		float val = 0.0;
        	for(size_t b = 0; b < B; b++){
                	for(size_t t = 0; t < T; t++){
				val += dweight_bt[(((b*T) + t) * C) + i];
			}
		}
		dweight[i] += val;
	}


	free(dweight_bt);

        return 0;
}


//this function implements the several types of Feed Forward Network, as requested by the specific model architecture
// "nl_type" stands for "the type of non-linearity used in this model"
//

// ============================================================
//  FFN activation functions
//
//  The non-linearity between the two linear layers of the
//  feed-forward network.
//  The attention layer routes information, but the
//  FFN transforms it. SiLU (Llama/Gemma), GELU (GPT-2), and
//  ReLU are supported. Gated variants (SwiGLU) multiply the
//  activation with a parallel linear projection.
// ============================================================

int ffn_io(int nl_type, byte * _out, byte * _in, byte * _in2, size_t io_size){

	float * out = (float *)_out;
	float * in  = (float *)_in;
	float * in2 = (float *)_in2;


	if(nl_type == NL_RELU){

		for(size_t i = 0; i < io_size; i++){

			float val = in[i];

			if(val<0.0)	val = 0.0;

			if(in2!=NULL)	val = val * in2[i];

			out[i] = val;
		}
	}
	else
	if(nl_type == NL_GELU_SIGMOID){

		//GeLU (x*sigmoid(1.7x))
		//sigmoid = (1/(1+expf(-x))
		//remember: derivative for sigmoid(x) along x: sigmoid(x)*(1-sigmoid(x)) (very efficient)

		for(size_t i = 0; i < io_size; i++){

			float val = in[i];

			val = val*(1.0/(1.0+expf(-1.7*val)));

			if(in2!=NULL)	val = val * in2[i];

			out[i] = val;
		}
	}
	else
	if(nl_type == NL_GELU_TANH){

		//GeLU approximated with TANH (sometimes used)
		for(size_t i = 0; i < io_size; i++){

			float val = in[i];

			val = 0.5 * val * (1.0 + tanhf(sqrtf(2.0/M_PI) * (val + (0.044715*val*val*val) )));
			
			if(in2!=NULL)	val = val * in2[i];

			out[i] = val;
		}
	}
	else
	if(nl_type == NL_SILU_LLAMA){

		//SILU as used in LLAMA2 model:  w2( silu(w1(x)) * w3(x) )

		for(size_t i = 0; i < io_size; i++){

			float val = in[i];	//in[] is "w1(x)" 

			val = val * (1.0/(1.0+expf(-1.0*val)));		//this is the computing of silu(w1(x))	

			val = val * in2[i];	//this is the peculiar part: in2[] is w3(x), and we perform "gating" of the silu by computing element-wise product of silu(w1(x)) and w3(x)

			out[i] = val;
		}
	}
	
}

//NOTE: can't use -Ofast here, because GELU gets broken. Thank you A.K. for the hint.
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif   
int ffn_io_backward(
	byte * _din, byte * _din2, 
	byte * _dout, byte *_in, byte *_in2, 
	int nl_type, size_t io_size
){

        float *  din = (float *)_din;   //original input
        float * din2 = (float *)_din2;	//gating input (if present)

	float * dout = (float *)_dout;
        float *   in = (float *)_in;    //original input
        float *  in2 = (float *)_in2;	//gating input (if present)


        if(nl_type == NL_RELU){

                for(size_t i = 0; i < io_size; i++){

                        float grad = dout[i];
                        float val = in[i];
                        float relu_val = (val < 0.0) ? 0.0 : val;
                        
                        //if in2 was used (multiplication), calculate gradient for in2 and adjust gradient for in
                        if(in2 != NULL){
                                if(din2 != NULL){
                                        din2[i] += grad * relu_val;
                                }
                                grad *= (val < 0.0) ? 0.0 : in2[i];
                        } else {
                                grad *= (val < 0.0) ? 0.0 : 1.0;
                        }
                        
                        din[i] += grad;
                }
        }
        else
        if(nl_type == NL_GELU_SIGMOID){

                //GeLU (x*sigmoid(1.7x))

                for(size_t i = 0; i < io_size; i++){

                        float x = in[i];
                        float sigmoid_val = 1.0 / (1.0 + expf(-1.7 * x));
                        float gelu_val = x * sigmoid_val;
                        
                        //derivative of GeLU sigmoid approximation
                        float sigmoid_derivative = sigmoid_val * (1.0 - sigmoid_val) * 1.7;
                        float gelu_derivative = sigmoid_val + x * sigmoid_derivative;
                        float grad = dout[i];
                        
                        //if in2 was used (multiplication), calculate gradient for in2 and adjust gradient for in
                        if(in2 != NULL){
                                if(din2 != NULL){
                                        din2[i] += grad * gelu_val;
                                }
                                grad *= gelu_derivative * in2[i];
                        } else {
                                grad *= gelu_derivative;
                        }
                        
                        din[i] += grad;
                }
        }
        else
        if(nl_type == NL_GELU_TANH){

                //GeLU approximated with TANH

                for(size_t i = 0; i < io_size; i++){
                        float x = in[i];
                        float cdf_term = sqrtf(2.0/M_PI) * (x + (0.044715 * x * x * x));
                        float tanh_val = tanhf(cdf_term);
                        float gelu_val = 0.5 * x * (1.0 + tanh_val);
                        
                        //derivative of GeLU tanh approximation
                        float dcdf_dx = sqrtf(2.0/M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
                        float dtanh_dcdf = 1.0 - tanh_val * tanh_val;
                        float dgelu_dx = 0.5 * (1.0 + tanh_val + x * dtanh_dcdf * dcdf_dx);
                        float grad = dout[i];
                        
                        //if in2 was used (multiplication), calculate gradient for in2 and adjust gradient for in
                        if(in2 != NULL){
                                if(din2 != NULL){
                                        din2[i] += grad * gelu_val;
                                }
                                grad *= dgelu_dx * in2[i];
                        } else {
                                grad *= dgelu_dx;
                        }
                        
                        din[i] += grad;
                }
        }
        else
        if(nl_type == NL_SILU_LLAMA){

                //SILU as used in LLAMA2 model: w2(silu(w1(x)) * w3(x))

                for(size_t i = 0; i < io_size; i++){
                        float x = in[i];  // in[] is "w1(x)"
                        float sigmoid_val = 1.0 / (1.0 + expf(-1.0 * x));
                        float silu_val = x * sigmoid_val;
                        
                        //derivative of SiLU (x*sigmoid(x)) with respect to x
                        float sigmoid_derivative = sigmoid_val * (1.0 - sigmoid_val);
                        float silu_derivative = sigmoid_val + x * sigmoid_derivative;
                        float grad = dout[i];
                        
                        //calculate gradient for in2 (w3(x))
                        if(din2 != NULL){
                                din2[i] += grad * silu_val;
                        }
                        
                        //calculate gradient for in (w1(x))
                        grad *= silu_derivative * in2[i];
                        
                        din[i] += grad;
                }
        }

        return 0;
}

#pragma float_control(pop)




