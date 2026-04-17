#define TRIP_MODEL_VERSION 2026041501
#include "trip.h"

// ============================================================
//  JSON config label mapping and config parsing
// ============================================================

char * json_multilabels[] = {

	//CONFIGURATION LABELS
	"architectures:       -",

	//vision model extra configuration (encoder)
	"vision_image_tokens:          num_image_tokens",
	"vision_patch_size:            patch_size",
	"target_dim_stream:            projection_dim",

	//model configuration (decoder or encoder)
	"dim_stream:          hidden_size, hidden_dim",
	"ffn_hidden_dim:      intermediate_size, intermediate_dim",
	"n_layers:            num_layers, num_hidden_layers",
	"n_queries:           num_query_heads, num_attention_heads",
	"n_keys:              num_key_value_heads",
	"vocab_size:          vocabulary_size, vocab_size",
	"sequence_maxtokens:  max_position_embeddings",
	"embeddings_cfg:      tie_word_embeddings",	//tied embeddings?
	"pose_cfg:            -",
	"norm_cfg[0]:         -",	//pre-attention normalization
	"norm_cfg[1]:         -",	//post-attention normalization
	"norm_cfg[2]:         -",	//final normalization
	"ffn_nl_type[0]:      hidden_act, projector_hidden_act",
	"ffn_nl_type[1]:      -",
	"bias_cfg[0]:         -",
	"bias_cfg[1]:         -",
	"bias_cfg[2]:         -",
	"bias_cfg[3]:         -",
	"norm_eps:            rms_norm_eps",
	"pose_theta:          rope_theta",


	//TRAINING ARGUMENTS LABELS
	"training_learning_rate:     learning_rate, lr",
	"training_lr_scheduler_type: lr_scheduler_type, lr_scheduler",	//used for cosine annealing learning rate
	"training_warmup_steps:      lr_warmup_steps, warmup_steps",	//used for cosine annealing learning rate
	"training_min_lr:            lr_min, min_lr",			//used for cosine annealing learning rate
	"training_beta1:         beta1, adam_beta1",
	"training_beta2:         beta2, adam_beta2",
	"training_epsilon:       epsilon, adam_epsilon",
	"training_weight_decay:  weight_decay",
	"training_batch_size:    batch_size, per_device_train_batch_size",
	"training_max_steps:     max_steps, max_train_steps",
	"training_num_epochs:    num_train_epochs, epochs",
	"training_save_steps:    save_steps, checkpointing_steps",
	"training_log_level:     log_level, logging_level",

	
	//TENSORS LABELS

	//extra tensors for vision model (encoder)
	"vision_embeddings_w:        patch_embedding.weight",
	"vision_embeddings_b:        patch_embedding.bias",
	"learned_pose_w:             position_embedding.weight",

	"multimodal_projector_w:     linear.weight",
	"multimodal_projector_b:     linear.bias",



	//tensors for language model (decoder)
	"embeddings:          embed_tokens.weight",
	"norm_pre_w:          layers.%d.input_layernorm.weight, layers.%d.layer_norm1.weight",
	"norm_pre_b:          layers.%d.input_layernorm.bias,   layers.%d.layer_norm1.bias",
	"qm:                  layers.%d.self_attn.q_proj.weight",
	"qb:                  layers.%d.self_attn.q_proj.bias",
	"km:                  layers.%d.self_attn.k_proj.weight",
	"kb:                  layers.%d.self_attn.k_proj.bias",
	"vm:                  layers.%d.self_attn.v_proj.weight",
	"vb:                  layers.%d.self_attn.v_proj.bias",
	"om:                  layers.%d.self_attn.o_proj.weight, layers.%d.self_attn.out_proj.weight",
	"ob:                  layers.%d.self_attn.o_proj.bias,   layers.%d.self_attn.out_proj.bias",
	"norm_post_w:         layers.%d.post_attention_layernorm.weight, layers.%d.layer_norm2.weight",
	"norm_post_b:         layers.%d.post_attention_layernorm.bias,   layers.%d.layer_norm2.bias",
	"pre_ffn_w:           layers.%d.mlp.gate_proj.weight, layers.%d.mlp.fc1.weight",
	"pre_ffn_b:           layers.%d.mlp.gate_proj.bias,   layers.%d.mlp.fc1.bias",
	"pre_ffn_w2:          layers.%d.mlp.up_proj.weight",
	"post_ffn_w:          layers.%d.mlp.down_proj.weight, layers.%d.mlp.fc2.weight",
	"post_ffn_b:          layers.%d.mlp.down_proj.bias,   layers.%d.mlp.fc2.bias",
	"norm_final_w:        norm.weight, post_layernorm.weight",
	"norm_final_b:        norm.bias,   post_layernorm.bias",
	"logits_classifier:   lm_head.weight"
};


JsonNode * scanJson_multilabel(JsonNode * root, char * key_label, char * prefix, int layer){

	JsonNode * ret = NULL;
	int n_multilabels = sizeof(json_multilabels)/sizeof(json_multilabels[0]);

	char labelbuf[1024];
	char temp_labelbuf[1024];

	int label_len;
	strcpy(labelbuf,key_label);
	strcat(labelbuf,":");	//the key label in TRiP's internal format for multi-labels is always followed by ":"
	label_len = strlen(labelbuf);

	for(int i=0; i<n_multilabels; i++){

		if(	(strlen(json_multilabels[i]) >= label_len)		//the compared entry must have sufficient length to compare!
			&& 
			(memcmp(labelbuf, json_multilabels[i], label_len)==0)	//key label found: let's look for the possible names in the json
		){
			
//char * p = json_multilabels[i] + label_len;
			char * p = json_multilabels[i];

			int len  = strlen(p);

			//first, let's extract the possible sub-labels from the label list referenced by the key
			while(len>0){
				//trim away spaces&co.
				while((len>0) && (*p<=' ')){
					p++;	len--;
				}
				int thislen = 0;

//while((len>0) && (*p!=',') && (*p>' '))
				while((len>0) && (*p!=',') && (*p>' ') && (*p!=':'))
				{
					p++;	len--;	thislen++;
				}
				if(thislen>0){	//an entire name has been extracted by the list: let's look for it in the json

					if(layer == -1){	//if layer number is not significant for this label
						memcpy(temp_labelbuf, p-thislen, thislen);
						temp_labelbuf[thislen] = '\0';
					}
					else{
						char template[1024];	//will be something like "model.layers.%d.something.weight"
						memcpy(template, p-thislen, thislen);
						template[thislen] = '\0';
						sprintf(temp_labelbuf, template, layer);	
					}

					sprintf(labelbuf,"%s%s",prefix,temp_labelbuf);



					mylog(LOG_VERBOSE_DEBUG, "   looking for possible name \"%s\" as a key in the json for label \"%s\"...",labelbuf,key_label);
					
	    				ret = findJsonNodeByKey(root, labelbuf);

					if(ret!=NULL){
						mylog(LOG_VERBOSE_DEBUG, "      ...key \"%s\" FOUND in the json!",labelbuf);
						return ret;
					}
					else{
						mylog(LOG_VERBOSE_DEBUG, "      ...key \"%s\" NOT found in the json.",labelbuf);
					}

				}
				
				p++;	len--;	
			}

		}
	}

	return ret;	
}



int jsonstring_to_tripint(char * in){
	char buf[256];
	strcpy(buf,in);
	int len = strlen(buf);
	for(int i=0;i<len;i++){
		if((buf[i]>='a') && (buf[i]<='z'))	buf[i]&=0xDF;	//lowercase to uppercase
	}

	if(strcmp(buf,"RELU")==0)		return 1;
	if(strcmp(buf,"GELU_SIGMOID")==0)	return 2;

	if(strcmp(buf,"GELU_TANH")==0)		return 3;
	if(strcmp(buf,"GELU_FAST")==0)		return 3;	//GELU_FAST refers to GELU_TANH, which is computationally less expensive
	if(strcmp(buf,"GELU")==0)		return 3;	//usually, GELU refers to GELU_TANH, which is computationally less expensive

	if(strcmp(buf,"SILU")==0)		return 4;

	if(strcmp(buf,"BASIC")==0)   		return 0;
	if(strcmp(buf,"LLAMAFORCAUSALLM")==0)   return 1;
	if(strcmp(buf,"GEMMAFORCAUSALLM")==0)   return 2;
	if(strcmp(buf,"PALIGEMMAFORCONDITIONALGENERATION")==0)   return 3;

	return atoi(buf);
}

float jsonstring_to_tripfloat(char * in){

	float ret;

	if(sscanf(in,"%f",&ret) != 1)	ret = 0.0;

	return ret;
}



JsonNode * json_get(JsonNode * root, char * file_text, char * TRIP_label, int TRIP_type, void * TRIP_target, int layer, char * default_value){

	char defbuf[4096];
	int using_default = 0; 
	JsonNode * jn;

	jn = scanJson_multilabel(root, TRIP_label, "", layer);

	if((jn==NULL) && (default_value==NULL)){
		mylog(LOG_ERROR,"Cannot find any key related to \"%s\" in %s file. Exiting...", TRIP_label, file_text);
		exit(-1);
	}
	else
	if((jn==NULL) && (default_value!=NULL)){

		mylog(LOG_INFO,"Cannot find any key related to \"%s\" in %s file. Using default value \"%s\"...", TRIP_label, file_text, default_value);

		using_default = 1;	

		sprintf(defbuf,"{ \"%s\": %s }", TRIP_label, default_value);
		char * p = defbuf;
		root = parseJsonValue(&p);
		jn = scanJson_multilabel(root, TRIP_label, "", layer);
		
	}
	else{
		mylog(LOG_VERBOSE_DEBUG,"Key \"%s\" found for \"%s\" in %s file.", jn->key, TRIP_label, file_text);
        }


	if(TRIP_type == TRIPTYPE_INT){

		//WILL MANAGE FIRST ELEMENT ONLY!!!		
		if(jn->type==JSON_ARRAY){
			jn = jn->value.children;
		}


		if((jn->type!=JSON_NUMBER) && (jn->type!=JSON_STRING) && (jn->type!=JSON_BOOL)){
			mylog(LOG_ERROR,"Key \"%s\" has invalid json type %s for \"%s\". Exiting...", jn->key, jsontype_text[jn->type], TRIP_label);
			exit(-1);
		} 

		*((int *)TRIP_target) = ( (jn->type==JSON_NUMBER) ? ((int)(jn->value.numberValue))                  :
		                          (jn->type==JSON_STRING) ? (jsonstring_to_tripint(jn->value.stringValue))  :
		                          (jn->type==JSON_BOOL)   ? (jn->value.boolValue)                           :  0  );

		mylog(LOG_VERBOSE_DEBUG,"\"%s\" = %d", TRIP_label, *((int *)TRIP_target));
	}
	else
	if(TRIP_type == TRIPTYPE_FLOAT){
		if((jn->type!=JSON_NUMBER) && (jn->type!=JSON_STRING)){
			mylog(LOG_ERROR,"Key \"%s\" has invalid json type %s for \"%s\". Exiting...", jn->key, jsontype_text[jn->type], TRIP_label);
			exit(-1);
		} 

		*((float *)TRIP_target) = ( (jn->type==JSON_NUMBER) ? ((float)(jn->value.numberValue))                  :
		                            (jn->type==JSON_STRING) ? (jsonstring_to_tripfloat(jn->value.stringValue))  :  0.0  );

		mylog(LOG_VERBOSE_DEBUG,"\"%s\" = %d", TRIP_label, *((float *)TRIP_target));
	}
	else{
		mylog(LOG_ERROR,"json_get: TRIP_type %d not handled. Exiting...", TRIP_type);
		exit(-1);
	}



	if(using_default==1)	freeJsonTree(root);

	return jn;
}




// ============================================================
//  Checkpoint file handling
// ============================================================

byte * safetensors_get_tensor(Model * model, char * key_label, int layer, char * safetensors_prefix){

	byte * tensor = NULL;
	JsonNode * tensors_map = NULL;

	for(int i=0; i < model->nfiles ; i++){

		char * json_pointer = model->header_data[i] + 8;
	
		tensors_map = parseJsonValue(&json_pointer);

		if(log_cfg >= LOG_DEBUG){
			mylog(LOG_VERBOSE_DEBUG,"scanning header %d for key label \"%s\"...",i,key_label);
			//printJsonTree(tensors_map, 0);
		}
	
		JsonNode * tensor_descr = scanJson_multilabel(tensors_map, key_label, safetensors_prefix, layer);

		if(tensor_descr!=NULL){
	
			mylog(LOG_VERBOSE_DEBUG,"Key label %s found in file #%d!", key_label, i);
			printJsonNode(tensor_descr, 0);

			JsonNode * ret = NULL;

			ret = findJsonNodeByKey(tensor_descr, "dtype");
			if(ret!=NULL){
				int this_wtype;
				if(strcmp(ret->value.stringValue,"BF16")==0)	this_wtype = WTYPE_BF16;
				else
				if(strcmp(ret->value.stringValue,"F16")==0)	this_wtype = WTYPE_FLOAT16;
				else
				if(strcmp(ret->value.stringValue,"F32")==0)	this_wtype = WTYPE_FLOAT32;
				else						this_wtype = WTYPE_UNDEFINED;


				if(this_wtype == WTYPE_UNDEFINED){
					mylog(LOG_ERROR,"unknown dtype \"%s\". This must be managed, impossible to continue. Exiting...",ret->value.stringValue);
					exit(1);
				}


				if(wtype == WTYPE_UNDEFINED){
					mylog(LOG_VERBOSE_INFO,"dtype \"%s\" just learned as model dtype.",ret->value.stringValue);
					wtype = this_wtype;
					wsize = wtype_bytesize[wtype];
				}
				else
				if(wtype != this_wtype){
					mylog(LOG_ERROR,"dtype \"%s\" mismatching current dtype \"%s\". TRiP currently does not support mixed weights. Exiting...",
						ret->value.stringValue, wtype_text[wtype]
					);
					exit(1);
				}
			}
			else{
				//do nothing: we assume that the dtype is unchanged
				mylog(LOG_VERBOSE_INFO,"missing dtype for this tensor; assuming model dtype \"%s\".",ret->value.stringValue);
			}

	
			ret = findJsonNodeByKey(tensor_descr, "data_offsets");
			if(ret!=NULL){
				size_t tensor_offset = atol((ret->value.children)->value.stringValue);
				tensor = model->tensors_data[i] + tensor_offset;
				mylog(LOG_VERBOSE_DEBUG,"OK: offset %zd found for tensor \"%s\".", tensor_offset, key_label);
			}
			else{
				mylog(LOG_ERROR,"missing offsets for tensor \"%s\". Safetensors file %d/%d is corrupt. Exiting...", i+1, model->nfiles);
				exit(1);
			}

			freeJsonTree(tensors_map);
			tensors_map = NULL;
			break;

		}
		else{

			mylog(LOG_VERBOSE_DEBUG,"Key label %s NOT found in file #%d!", key_label, i);
		}
	}


	if(tensors_map!=NULL)	freeJsonTree(tensors_map);

	return tensor;
}




size_t get_checkpointfile_headersize(Model * model, int whichfile){

	size_t s;

	if(checkpoint_type == CP_UNDEFINED){		//error!

		mylog(LOG_ERROR,"Checkpoint type UNDEFINED! Exiting...");
		exit(-1);
	}
	else
	if(checkpoint_type == CP_LLAMA2_AK){	//checkpoint file for LLAMA2 C implementation from A.Karpathy
						
		s = 7 * sizeof(int);
	}
	else
	if(checkpoint_type == CP_GPT2_AK){	//checkpoint file for GPT2 C implementation from A.Karpathy
						
		s = 256 * sizeof(int);
	}
	else
	if(checkpoint_type == CP_SAFETENSORS){	//checkpoint file in SafeTensors format
		uint64_t json_length;
		memcpy(&json_length, model->file_data[whichfile],8);
		s = 8 + json_length;
	}


	return s;
}

//helper function to generate a random number from a standard normal distribution (mean 0, stddev 1)
//(this uses the "Box-Muller" transform, which goes from uniform distribution to normal distribution)

// ============================================================
//  Weight initialization
// ============================================================

//helper function to generate a random number from a standard normal distribution (mean 0, stddev 1)
//(this uses the "Box-Muller" transform, which goes from uniform distribution to normal distribution)
float random_normal(){
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * M_PI * u2);
}

//initializes a tensor with values from a normal distribution
void initialize_tensor_normal(float * tensor, size_t num_elements, float mean, float stddev){
    size_t i;
    //#pragma omp parallel for private(i)
    for(i = 0; i < num_elements; i++){
        tensor[i] = mean + (stddev * random_normal());
    }
}

//initializes a tensor using Xavier/Glorot normal initialization
void initialize_tensor_xavier(float * tensor, size_t num_elements, size_t fan_in, size_t fan_out){
    float stddev = sqrtf(2.0 / (fan_in + fan_out));
    initialize_tensor_normal(tensor, num_elements, 0.0, stddev);
}

//initializes a tensor using He/Kaiming normal initialization
void initialize_tensor_he(float * tensor, size_t num_elements, size_t fan_in){
    float stddev = sqrtf(2.0 / fan_in);
    initialize_tensor_normal(tensor, num_elements, 0.0, stddev);
}

//initializes a tensor with a constant value
void initialize_tensor_constant(float * tensor, size_t num_elements, float value){
    //#pragma omp parallel for
    for(size_t i = 0; i < num_elements; i++){
        tensor[i] = value;
    }
}




void init_weights(Model * model){

    mylog(LOG_INFO, "Initializing model weights from configuration...");

    //set the weight type to float32 for initialization
    wtype = WTYPE_FLOAT32;
    wsize = wtype_bytesize[wtype];

    //we love short names
    size_t dim_stream = model->config.dim_stream;
    size_t hidden_dim = model->config.ffn_hidden_dim;
    size_t vocab_size = model->config.vocab_size;
    size_t n_layers   = model->config.n_layers;
    size_t n_queries  = model->config.n_queries;
    size_t n_keys     = model->config.n_keys;
    size_t dim_qkv    = dim_stream / n_queries;

    //before we start, let's allocate the pointers to the arrays for PER-LAYER weights
    model->w.norm_pre_w		= myalloc(n_layers * sizeof(byte *));
    model->w.norm_pre_b		= myalloc(n_layers * sizeof(byte *));
    model->w.qm			= myalloc(n_layers * sizeof(byte *));
    model->w.km			= myalloc(n_layers * sizeof(byte *));
    model->w.vm			= myalloc(n_layers * sizeof(byte *));
    model->w.om			= myalloc(n_layers * sizeof(byte *));
    model->w.qb			= myalloc(n_layers * sizeof(byte *));
    model->w.kb			= myalloc(n_layers * sizeof(byte *));
    model->w.vb			= myalloc(n_layers * sizeof(byte *));
    model->w.ob			= myalloc(n_layers * sizeof(byte *));
    model->w.norm_post_w	= myalloc(n_layers * sizeof(byte *));
    model->w.norm_post_b	= myalloc(n_layers * sizeof(byte *));
    model->w.pre_ffn_w		= myalloc(n_layers * sizeof(byte *));
    model->w.pre_ffn_b		= myalloc(n_layers * sizeof(byte *));
    model->w.pre_ffn_w2		= myalloc(n_layers * sizeof(byte *));
    model->w.post_ffn_w		= myalloc(n_layers * sizeof(byte *));
    model->w.post_ffn_b		= myalloc(n_layers * sizeof(byte *));



    //let's initialize...

    //EMBEDDINGS
    size_t embedding_size = vocab_size * dim_stream;
    model->w.embeddings = myalloc(embedding_size * wsize);
    initialize_tensor_normal((float *)model->w.embeddings, embedding_size, 0.0, 0.02);
    mylog(LOG_VERBOSE_INFO, "Initialized token embeddings: N(0, 0.02)");

    //LOGITS CLASSIFIER
    if(model->config.embeddings_cfg == EMBEDDINGS_SHARED){	//if shared
        model->w.logits_classifier = model->w.embeddings;
    }
    else{							//if NOT shared
        model->w.logits_classifier = myalloc(embedding_size * wsize);
        initialize_tensor_normal((float *)model->w.logits_classifier, embedding_size, 0.0, 0.02);
        mylog(LOG_VERBOSE_INFO, "Initialized logits classifier: N(0, 0.02)");
    }


    //let's initialize per-layer weights
    for(size_t l = 0; l < n_layers; l++){


	//ATTENTION LAYERS

        //attention Q,K,V,O projections: Xavier init
        model->w.qm[l] = myalloc(dim_stream * n_queries * dim_qkv * wsize);
        initialize_tensor_xavier((float *)model->w.qm[l], (dim_stream * n_queries * dim_qkv), dim_stream, (n_queries * dim_qkv));

        model->w.km[l] = myalloc(dim_stream * n_keys * dim_qkv * wsize);
        initialize_tensor_xavier((float *)model->w.km[l], (dim_stream * n_keys * dim_qkv)   , dim_stream, (n_keys    * dim_qkv));
        
        model->w.vm[l] = myalloc(dim_stream * n_keys * dim_qkv * wsize);
        initialize_tensor_xavier((float *)model->w.vm[l], (dim_stream * n_keys * dim_qkv)   , dim_stream, (n_keys    * dim_qkv));
        
        model->w.om[l] = myalloc(n_queries * dim_qkv * dim_stream * wsize);
        initialize_tensor_xavier((float *)model->w.om[l], (n_queries * dim_qkv * dim_stream), (n_queries * dim_qkv), dim_stream);
        mylog(LOG_VERBOSE_INFO, "layer %zu: Q,K,V,O matrices initialized with Xavier method", l);

        //attention biases (init to 0)
        if(model->config.bias_cfg[3] == BIAS_ON){

            model->w.qb[l] = myalloc(n_queries * dim_qkv * wsize);
            initialize_tensor_constant((float *)model->w.qb[l], n_queries * dim_qkv, 0.0);
            model->w.kb[l] = myalloc(n_keys * dim_qkv * wsize);
            initialize_tensor_constant((float *)model->w.kb[l], n_keys * dim_qkv   , 0.0);
            model->w.vb[l] = myalloc(n_keys * dim_qkv * wsize);
            initialize_tensor_constant((float *)model->w.vb[l], n_keys * dim_qkv   , 0.0);
            model->w.ob[l] = myalloc(dim_stream * wsize);
            initialize_tensor_constant((float *)model->w.ob[l], dim_stream         , 0.0);
            mylog(LOG_VERBOSE_INFO, "layer %zu: attention biases initialized to 0.0", l);
        }




        //FFN LAYERS 

        //pre-FFN (linear1 / gate_proj)
        model->w.pre_ffn_w[l] = myalloc(dim_stream * hidden_dim * wsize);

        if(model->config.ffn_nl_type[0] == NL_RELU){

            initialize_tensor_he((float *)model->w.pre_ffn_w[l], (dim_stream * hidden_dim), dim_stream);
            mylog(LOG_VERBOSE_INFO, "layer %zu: pre_ffn_w initialized with He (reason: activation function is ReLU)", l);
        }
	else{

            initialize_tensor_xavier((float *)model->w.pre_ffn_w[l], (dim_stream * hidden_dim), dim_stream, hidden_dim);
            mylog(LOG_VERBOSE_INFO, "layer %zu: pre_ffn_w initialized with Xavier", l);
        }


        if(model->config.bias_cfg[0] == BIAS_ON){

            model->w.pre_ffn_b[l] = myalloc(hidden_dim * wsize);
            initialize_tensor_constant((float *)model->w.pre_ffn_b[l], hidden_dim, 0.0);
            mylog(LOG_VERBOSE_INFO, "layer %zu: pre_ffn_b initialized to 0.0", l);
        }
        
        //gated FFN (up_proj) if applicable
        if(model->config.ffn_nl_type[1] == GATE_ON){

            model->w.pre_ffn_w2[l] = myalloc(dim_stream * hidden_dim * wsize);
            initialize_tensor_xavier((float *)model->w.pre_ffn_w2[l], (dim_stream * hidden_dim), dim_stream, hidden_dim);
            mylog(LOG_VERBOSE_INFO, "layer %zu: pre_ffn_w2 (gate) initialized with Xavier", l);
        }

        //post-FFN (linear2 / down_proj)
        model->w.post_ffn_w[l] = myalloc(hidden_dim * dim_stream * wsize);
        initialize_tensor_xavier((float *)model->w.post_ffn_w[l], hidden_dim * dim_stream, hidden_dim, dim_stream);
        mylog(LOG_VERBOSE_INFO, "layer %zu: post_ffn_w initialiized with Xavier", l);

        if(model->config.bias_cfg[1] == BIAS_ON){

            model->w.post_ffn_b[l] = myalloc(dim_stream * wsize);
            initialize_tensor_constant((float *)model->w.post_ffn_b[l], dim_stream, 0.0);
        }



        //NORMALIZATION LAYERS
        if(model->config.norm_cfg[0] != NORM_NONE){ //pre-attention norm

            model->w.norm_pre_w[l] = myalloc(dim_stream * wsize);
            initialize_tensor_constant((float *)model->w.norm_pre_w[l], dim_stream, 1.0);
            mylog(LOG_VERBOSE_INFO, "layer %zu: norm_pre_w initialized to 1.0", l);

            if(model->config.norm_cfg[0] == NORM_LAYERNORM){

                model->w.norm_pre_b[l] = myalloc(dim_stream * wsize);
                initialize_tensor_constant((float *)model->w.norm_pre_b[l], dim_stream, 0.0);
            	mylog(LOG_VERBOSE_INFO, "layer %zu: norm_pre_b initialized to 0.0", l);
            }
        }

        if(model->config.norm_cfg[1] != NORM_NONE){ //post-attention norm

            model->w.norm_post_w[l] = myalloc(dim_stream * wsize);
            initialize_tensor_constant((float *)model->w.norm_post_w[l], dim_stream, 1.0);
            mylog(LOG_VERBOSE_INFO, "layer %zu: norm_post_w initialized to 1.0", l);

            if(model->config.norm_cfg[1] == NORM_LAYERNORM){

                model->w.norm_post_b[l] = myalloc(dim_stream * wsize);
                initialize_tensor_constant((float *)model->w.norm_post_b[l], dim_stream, 0.0);
            	mylog(LOG_VERBOSE_INFO, "layer %zu: norm_post_b initialized to 0.0", l);
            }
        }
    }



    //final norm
    if(model->config.norm_cfg[2] != NORM_NONE){

        model->w.norm_final_w = myalloc(dim_stream * wsize);
        initialize_tensor_constant((float *)model->w.norm_final_w, dim_stream, 1.0);
            mylog(LOG_VERBOSE_INFO, "layer %zu: norm_final_w initialized to 1.0");


        if (model->config.norm_cfg[2] == NORM_LAYERNORM) {
            model->w.norm_final_b = myalloc(dim_stream * wsize);
            initialize_tensor_constant((float *)model->w.norm_final_b, dim_stream, 0.0f);
            mylog(LOG_VERBOSE_INFO, "layer %zu: norm_final_b initialized to 0.0");
        }
    }



    mylog(LOG_INFO, "Model weights initialization complete!");
}






// ============================================================
//  Training configuration
// ============================================================

void load_training_args(char * filename){

    FILE * f;
    size_t size;
    char * json_data;
    JsonNode * root = NULL;


    //1) read the JSON file
    if((f = fopen(filename, "rb")) == NULL) {
        mylog(LOG_ERROR, "Training configuration file \"%s\" not found. Exiting...", filename);
        exit(-1);
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    json_data = (char *)malloc(size + 1);
    if(fread(json_data, 1, size, f) != size) {
        mylog(LOG_ERROR, "Cannot read training configuration file \"%s\". Exiting...", filename);
        fclose(f);
        free(json_data);
        exit(-1);
    }
    fclose(f);
    json_data[size] = '\0';


    //2) parse the JSON content (i.e.: the training configuration)
    char * json_pointer = json_data;
    root = parseJsonValue(&json_pointer);
    if(root == NULL){
        mylog(LOG_ERROR, "Failed to parse JSON from training config file \"%s\". Exiting...", filename);
        free(json_data);
        exit(-1);
    }
    free(json_data); //free the raw JSON data buffer


    mylog(LOG_INFO, "Loading training configuration from \"%s\"...", filename);


    //3) extract training arguments and update engine configuration
    float temp_lr, temp_b1, temp_b2, temp_eps, temp_wd;
    int temp_batch, temp_steps, temp_epochs, temp_save;

    //AdamW (optimizer) parameters
    json_get(root, filename, "training_learning_rate", TRIPTYPE_FLOAT, &temp_lr,  -1, "5e-6");

    //get LR Scheduler settings
    JsonNode * scheduler_node = scanJson_multilabel(root, "training_lr_scheduler_type", "", -1);
    if((scheduler_node != NULL)  &&  (scheduler_node->type == JSON_STRING)){
        strncpy(training_lr_scheduler_type, scheduler_node->value.stringValue, sizeof(training_lr_scheduler_type) - 1);
    }
    
    //if scheduler is "cosine" (cosine annealing learning rate), let's get its specific parameters
    if(strcmp(training_lr_scheduler_type, "cosine") == 0){
        int temp_warmup;
        float temp_min_lr_float;
        json_get(root, filename, "training_warmup_steps", TRIPTYPE_INT, &temp_warmup, -1, "0");
        json_get(root, filename, "training_min_lr", TRIPTYPE_FLOAT, &temp_min_lr_float, -1, "0.0");
        training_warmup_steps = (ssize_t)temp_warmup;
        training_min_lr = temp_min_lr_float;
        mylog(LOG_INFO, "Using Cosine Annealing LR scheduler (warmup: %zd steps, min_lr: %.2e)", training_warmup_steps, training_min_lr);
    }
    else{
        mylog(LOG_INFO, "Using Constant LR scheduler.");
    }



    json_get(root, filename, "training_beta1",         TRIPTYPE_FLOAT, &temp_b1,  -1, "0.9");
    json_get(root, filename, "training_beta2",         TRIPTYPE_FLOAT, &temp_b2,  -1, "0.999");
    json_get(root, filename, "training_epsilon",       TRIPTYPE_FLOAT, &temp_eps, -1, "1e-8");
    json_get(root, filename, "training_weight_decay",  TRIPTYPE_FLOAT, &temp_wd,  -1, "0.0");

    //LET'S APPLY THE TRAINING PARAMETERS TO THE OPTIMIZER!
    adamw_set_config(temp_lr, temp_b1, temp_b2, temp_eps, temp_wd);

    mylog(LOG_INFO, "AdamW config:    learning_rate = %.2e    beta1=%.2f    beta2=%.3f    eps=%.2e    weight_decay=%.2f",
          adamw_cfg.learning_rate, adamw_cfg.beta1, adamw_cfg.beta2, adamw_cfg.epsilon, adamw_cfg.weight_decay
    );


    //training control parameters
    json_get(root, filename, "training_batch_size",    TRIPTYPE_INT, &temp_batch,  -1, "1");
    json_get(root, filename, "training_max_steps",     TRIPTYPE_INT, &temp_steps,  -1, "-1");
    json_get(root, filename, "training_num_epochs",    TRIPTYPE_INT, &temp_epochs, -1, "1");
    json_get(root, filename, "training_save_steps",    TRIPTYPE_INT, &temp_save,   -1, "0");
    training_batch_size = (ssize_t)temp_batch;
    training_max_steps  = (ssize_t)temp_steps;
    training_num_epochs = (ssize_t)temp_epochs;
    training_save_steps = (ssize_t)temp_save;

    char txt1[256];
    char txt2[256];
    if(training_max_steps <= 0)		sprintf(txt1,"OFF (%zd)",training_max_steps);
    else				sprintf(txt1,"%zd"      ,training_max_steps);
    if(training_save_steps <= 0)	sprintf(txt2,"OFF (%zd)",training_save_steps);
    else				sprintf(txt2,"%zd"      ,training_save_steps);


    mylog(LOG_INFO, "General training configuration:    batch_size = %zu    max_steps = %s    num_epochs = %zu    save_steps = %s",
          training_batch_size, txt1, training_num_epochs, txt2
    );


    //logging level
    JsonNode * log_node = scanJson_multilabel(root, "training_log_level", "", -1);

    if((log_node != NULL)  &&  (log_node->type == JSON_STRING)){

        char * level_str = log_node->value.stringValue;
        //convert string to uppercase for case-insensitive comparison
        for(int i = 0; (level_str[i]!='\0'); i++) { level_str[i] = toupper(level_str[i]); }

	log_cfg = LOG_INFO;	//default

	if(strcmp(level_str, "VERBOSE_INFO") == 0) 	log_cfg = LOG_VERBOSE_INFO;
        else
	if(strcmp(level_str, "DEBUG") == 0) 		log_cfg = LOG_DEBUG;
        else 
	if(strcmp(level_str, "VERBOSE_DEBUG") == 0) 	log_cfg = LOG_VERBOSE_DEBUG;

        mylog(LOG_INFO, "LOG LEVEL = %s", level_str);
    }


    //clean-up
    freeJsonTree(root);
}




// ============================================================
//  Model load and save
// ============================================================

void load_model_from_checkpoint(Model * model, char * path){

	FILE * f;
	ssize_t size;
	int curr_n;


	//FIRST, let's handle the possibility of a checkpoint spread across multiple files
	model->dirpath = calloc((strlen(path)+1), sizeof(char));	//+1 is just to handle terminator
	int dirpath_len;

	//let's extrapolate the specified directory path (filename excluded)
	strcpy(model->dirpath,path);
	dirpath_len = strlen(model->dirpath);
	while(dirpath_len>=0){
		if(model->dirpath[dirpath_len] == '/'){
			model->dirpath[dirpath_len] = '\0';
			break;
		}
		dirpath_len--;
	}
	dirpath_len = strlen(model->dirpath);
		

	if(checkpoint_type == CP_SAFETENSORS){
	
		if(sscanf(path+dirpath_len+1,"model-%5d-of-%5d.safetensors",&curr_n,&model->nfiles) != 2){
			model->nfiles = 1;
		}
		mylog(LOG_DEBUG,"SafeTensors checkpoint: %d file(s) found.",model->nfiles);
	}
	else{
		model->nfiles = 1;
	}


	//let's allocate the arrays of: file descriptors, pointers to data, checkpoint file sizes
	model->fd =              calloc(model->nfiles, sizeof(int));
	model->file_data =       calloc(model->nfiles, sizeof(unsigned char *));
	model->header_data =     calloc(model->nfiles, sizeof(unsigned char *));
	model->tensors_data =    calloc(model->nfiles, sizeof(unsigned char *));
	model->checkpoint_size = calloc(model->nfiles, sizeof(ssize_t));


	//now, let's create all the "handles" to the files!
	char * currpath = calloc((strlen(path)+8), sizeof(char));	//+8 is just to handle terminator + any change to the string I may require in the future

	for(int i=0; i < model->nfiles; i++){

		if(model->nfiles > 1){
			if(checkpoint_type == CP_SAFETENSORS){
				sprintf(currpath,"%s/model-%05d-of-%05d.safetensors",model->dirpath,(i+1),model->nfiles);
			}
			else{
				mylog(LOG_ERROR,"Unhandled multiple files for checkpoint type \"%s\". Exiting...",cp_litteral[checkpoint_type]); exit(-1); 
			}
		}
		else{
			strcpy(currpath,path);	//if we have just one file, the path is the one specified by the user, no matter the checkpoint type
		}


		if((f = fopen(currpath,"rb")) == NULL){
			mylog(LOG_ERROR,"Cannot open checkpoint file \"%s\". Exiting...",currpath); exit(-1); 
		}
		fseek(f,0,SEEK_END);
		size = ftell(f);
		fclose(f);

		if(size<0){
			mylog(LOG_ERROR,"Checkpoint file \"%s\" size is invalid (<0). Exiting...",currpath); exit(-1); 
		}



		unsigned char * thisdata;

		model->fd[i] = open(currpath,O_RDONLY);


		if(model->fd[i] < 0){
			mylog(LOG_ERROR,"Cannot open checkpoint file \"%s\". Exiting...",currpath); exit(-1);
		}
	

		thisdata = mmap(NULL, size, PROT_READ, MAP_PRIVATE, model->fd[i], 0);

		if(thisdata == MAP_FAILED){
			mylog(LOG_ERROR,"Cannot mmap the checkpoint file \"%s\". Exiting...",currpath); exit(-1);
		}
		else{
			model->checkpoint_size[i] = size;
			model->file_data[i] = thisdata;
			mylog(LOG_INFO,"Checkpoint mmap successful from file \"%s\" (%zd bytes).",currpath,size);
		}
	}


	free(currpath);


	if(checkpoint_type == CP_GPT2_AK){
	
		if(memcmp(model->file_data[0], "\xc6\xd7\x34\x01\x03\x00\x00\x00", 8) != 0){

			mylog(LOG_ERROR,"Checkpoint file \"%s\" is not in GPT2_AK format (invalid header)! Exiting...",currpath); exit(-1);
		}
	}


	return;
}


//function to add tensor metadata to JSON
void add_tensor_metadata(const char * name, size_t * shape, int n_dims, size_t * offset, char * json_buffer, char * dtype_str){

        if(strlen(json_buffer) > 1)	strcat(json_buffer, ",");

        strcat(json_buffer, "\n\"");
        strcat(json_buffer, name);
        strcat(json_buffer, "\": {\"dtype\": \"");
        strcat(json_buffer, dtype_str);
        strcat(json_buffer, "\", \"shape\": [");
        
        size_t total_elements = 1;
        for(int i = 0; i < n_dims; i++){
            if(i > 0)	strcat(json_buffer, ", ");
            char num[32];
            sprintf(num, "%zu", shape[i]);
            strcat(json_buffer, num);
            total_elements *= shape[i];
        }
        
        strcat(json_buffer, "], \"data_offsets\": [");
        char offset_str[32];
        sprintf(offset_str, "%zu", *offset);
        strcat(json_buffer, offset_str);
        strcat(json_buffer, ", ");
        *offset += total_elements * target_wsize;
        sprintf(offset_str, "%zu", *offset);
        strcat(json_buffer, offset_str);
        strcat(json_buffer, "]}");
}
    
//function to convert and write tensor data
size_t convert_and_write_tensor(FILE* f, byte* src_data, size_t num_elements){


        //temporary buffer for weight conversion
	byte * temp_buffer = myalloc(num_elements * target_wsize);
 
        //convert data
        for(size_t i = 0; i < num_elements; i++){
            float val;
            
            //read from source
            if(wtype == WTYPE_FLOAT32){
                val = ((float*)src_data)[i];
            }
	    else
	    if(wtype == WTYPE_BF16){
                val = (float)(((__bf16*)src_data)[i]);
            }
	    else
	    if(wtype == WTYPE_FLOAT16){
                val = (float)(((_Float16*)src_data)[i]);
            }
            
            //write to target
            if(target_wtype == WTYPE_FLOAT32){
                ((float*)temp_buffer)[i] = val;
            }
	    else
	    if(target_wtype == WTYPE_BF16){
                (((__bf16*)temp_buffer)[i] = (__bf16)val);
            }
	    else
	    if(target_wtype == WTYPE_FLOAT16){
                (((_Float16*)temp_buffer)[i] = (_Float16)val);
            }
        }
        
	int ret;
        ret = fwrite(temp_buffer, target_wsize, num_elements, f);

	free(temp_buffer);

	return ret;
}
 
//function to convert and write tensor data, same as above, but with interleaved rows
size_t convert_and_write_tensor_interleaved(FILE* f, byte* src_data, size_t num_elements, size_t row_size){


        //temporary buffer for weight conversion
	byte * temp_buffer = myalloc(num_elements * target_wsize);
 
        //convert data: we loop on the index of the TARGET (interleaved) matrix
        for(size_t ix = 0; ix < num_elements; ix++){

	    size_t i;
	    size_t half_rows = (num_elements/row_size)/2;
	    size_t target_row = (ix/row_size);
	    size_t source_row = (target_row<half_rows) ? (target_row*2) : (((target_row-half_rows)*2)+1);

	    //this is the index we must use on the SOURCE (not interleaved) matrix
	    i = (source_row*row_size) + (ix%row_size);



            float val;
            
            //read from source
            if(wtype == WTYPE_FLOAT32){
                val = ((float*)src_data)[i];
            }
	    else
	    if(wtype == WTYPE_BF16){
                val = (float)(((__bf16*)src_data)[i]);
            }
	    else
	    if(wtype == WTYPE_FLOAT16){
                val = (float)(((_Float16*)src_data)[i]);
            }
            
            //write to target
            if(target_wtype == WTYPE_FLOAT32){
                ((float*)temp_buffer)[i] = val;
            }
	    else
	    if(target_wtype == WTYPE_BF16){
                (((__bf16*)temp_buffer)[i] = (__bf16)val);
            }
	    else
	    if(target_wtype == WTYPE_FLOAT16){
                (((_Float16*)temp_buffer)[i] = (_Float16)val);
            }
        }
        
	int ret;
        ret = fwrite(temp_buffer, target_wsize, num_elements, f);

	free(temp_buffer);

	return ret;
}
 
int save_model(Model * model){

    FILE * f;
    uint64_t json_length;
    size_t current_offset = 0;
    

	if(target_wtype == WTYPE_UNDEFINED){
		target_wtype = wtype;
	}

	// Get current date and time
	time_t rawtime;
	struct tm * timeinfo;
	char time_str[32];
	
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", timeinfo);

	char filename[256];
	sprintf(filename,"model.safetensors.%s",time_str);

	
    //open output file
    f = fopen(filename, "wb");
    if (f == NULL) {
        mylog(LOG_ERROR, "Cannot open file %s for writing. Exiting...",filename);
        return -1;
    }
    
    //create JSON metadata structure
    char * json_buffer = (char*)myalloc(10*1024*1024);	//let's say 10MB buffer for JSON header should be enough :D
    strcpy(json_buffer, "{");
    

    //helper variables
    size_t dim_qkv = (size_t)(model->config.dim_stream / model->config.n_queries);
    size_t dim_stream = (size_t)(model->config.dim_stream);
    size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
    size_t vocab_size = (size_t)(model->config.vocab_size);
    size_t n_layers = (size_t)(model->config.n_layers);
    size_t n_queries = (size_t)(model->config.n_queries);
    size_t n_keys = (size_t)(model->config.n_keys);
    size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
    
    char * dtype_str;

    switch(target_wtype){
        case WTYPE_FLOAT32:
            dtype_str = "F32";
            target_wsize = 4;
            break;
        case WTYPE_BF16:
            dtype_str = "BF16";
            target_wsize = 2;
            break;
        case WTYPE_FLOAT16:
            dtype_str = "F16";
            target_wsize = 2;
            break;
        default:
            mylog(LOG_ERROR, "Unsupported target weight type %d", target_wtype);
            fclose(f);
	    free(json_buffer);
            return -1;
    }
    
  



 
   
    //build JSON metadata for all tensors
    char tensor_name[512];
    char * prefix = "";
    
    if(model->config.submodel_type == MODELTYPE_VISION_ENCODER) {
        if (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL) {
            prefix = "vision_tower.vision_model.";
            
            //vision embeddings
            size_t shape1[] = {3 * model->config.vision_patch_size * model->config.vision_patch_size, dim_stream};
            add_tensor_metadata("vision_tower.vision_model.embeddings.patch_embedding.weight", shape1, 2, &current_offset, json_buffer, dtype_str);
            
            size_t shape2[] = {dim_stream};
            add_tensor_metadata("vision_tower.vision_model.embeddings.patch_embedding.bias", shape2, 1, &current_offset, json_buffer, dtype_str);
            
            size_t shape3[] = {model->config.vision_image_tokens, dim_stream};
            add_tensor_metadata("vision_tower.vision_model.embeddings.position_embedding.weight", shape3, 2, &current_offset, json_buffer, dtype_str);
            
            //multimodal projector
            size_t shape4[] = {dim_stream, model->config.target_dim_stream};
            add_tensor_metadata("multi_modal_projector.linear.weight", shape4, 2, &current_offset, json_buffer, dtype_str);
            
            size_t shape5[] = {model->config.target_dim_stream};
            add_tensor_metadata("multi_modal_projector.linear.bias", shape5, 1, &current_offset, json_buffer, dtype_str);
        }
    }
    else
    if(model->config.submodel_type == MODELTYPE_DECODER){

        if(model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL){
            prefix = "language_model.model.";
        }else
	if((model->config.architectures == ARCH_LLAMA_CAUSAL)  ||  (model->config.architectures == ARCH_GEMMA_CAUSAL)){
            prefix = "model.";
        }
        
        //embeddings
        sprintf(tensor_name, "%sembed_tokens.weight", prefix);
        size_t shape_emb[] = {vocab_size, dim_stream};
        add_tensor_metadata(tensor_name, shape_emb, 2, &current_offset, json_buffer, dtype_str);
        
        //logits classifier (if unshared)
        if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){
            size_t shape_lm[] = {vocab_size, dim_stream};
            add_tensor_metadata("lm_head.weight", shape_lm, 2, &current_offset, json_buffer, dtype_str);
        }
    }
    
    //per-layer tensors
    char layer_prefix[256];
    if((model->config.submodel_type == MODELTYPE_VISION_ENCODER)  &&  (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)){
        strcpy(layer_prefix, "vision_tower.vision_model.encoder.");
    }
    else{
        strcpy(layer_prefix, prefix);
    }
    
    for(size_t layer = 0; layer < n_layers; layer++){
        
        //pre-attention normalization
        if((model->config.norm_cfg[0] == NORM_LAYERNORM)  ||  (model->config.norm_cfg[0] == NORM_RMSNORM)){

            size_t shape[] = {dim_stream};

            if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
                sprintf(tensor_name, "%slayers.%zu.layer_norm1.weight", layer_prefix, layer);
            }
	    else{
                sprintf(tensor_name, "%slayers.%zu.input_layernorm.weight", layer_prefix, layer);
            }

            add_tensor_metadata(tensor_name, shape, 1, &current_offset, json_buffer, dtype_str);
            
            if(model->config.norm_cfg[0] == NORM_LAYERNORM){

                if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
                    sprintf(tensor_name, "%slayers.%zu.layer_norm1.bias", layer_prefix, layer);
                }
		else{
                    sprintf(tensor_name, "%slayers.%zu.input_layernorm.bias", layer_prefix, layer);
                }

                add_tensor_metadata(tensor_name, shape, 1, &current_offset, json_buffer, dtype_str);
            }
        }

        
        // Q, K, V, O projections
        size_t shape_q[] = {dim_stream, n_queries * dim_qkv};
        sprintf(tensor_name, "%slayers.%zu.self_attn.q_proj.weight", layer_prefix, layer);
        add_tensor_metadata(tensor_name, shape_q, 2, &current_offset, json_buffer, dtype_str);
        
        size_t shape_kv[] = {dim_stream, n_keys * dim_qkv};
        sprintf(tensor_name, "%slayers.%zu.self_attn.k_proj.weight", layer_prefix, layer);
        add_tensor_metadata(tensor_name, shape_kv, 2, &current_offset, json_buffer, dtype_str);
        
        sprintf(tensor_name, "%slayers.%zu.self_attn.v_proj.weight", layer_prefix, layer);
        add_tensor_metadata(tensor_name, shape_kv, 2, &current_offset, json_buffer, dtype_str);
        
        size_t shape_o[] = {n_queries * dim_qkv, dim_stream};
        if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
            sprintf(tensor_name, "%slayers.%zu.self_attn.out_proj.weight", layer_prefix, layer);
        }
	else{
            sprintf(tensor_name, "%slayers.%zu.self_attn.o_proj.weight", layer_prefix, layer);
        }
        add_tensor_metadata(tensor_name, shape_o, 2, &current_offset, json_buffer, dtype_str);
        

        //biases if present
        if(model->config.bias_cfg[3] == BIAS_ON){

            size_t shape_qb[] = {n_queries * dim_qkv};
            sprintf(tensor_name, "%slayers.%zu.self_attn.q_proj.bias", layer_prefix, layer);
            add_tensor_metadata(tensor_name, shape_qb, 1, &current_offset, json_buffer, dtype_str);
            
            size_t shape_kvb[] = {n_keys * dim_qkv};
            sprintf(tensor_name, "%slayers.%zu.self_attn.k_proj.bias", layer_prefix, layer);
            add_tensor_metadata(tensor_name, shape_kvb, 1, &current_offset, json_buffer, dtype_str);
            
            sprintf(tensor_name, "%slayers.%zu.self_attn.v_proj.bias", layer_prefix, layer);
            add_tensor_metadata(tensor_name, shape_kvb, 1, &current_offset, json_buffer, dtype_str);
            
            size_t shape_ob[] = {dim_stream};
            if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
                sprintf(tensor_name, "%slayers.%zu.self_attn.out_proj.bias", layer_prefix, layer);
            }
	    else{
                sprintf(tensor_name, "%slayers.%zu.self_attn.o_proj.bias", layer_prefix, layer);
            }
            add_tensor_metadata(tensor_name, shape_ob, 1, &current_offset, json_buffer, dtype_str);
        }
        
        //post-attention normalization
        if((model->config.norm_cfg[1] == NORM_LAYERNORM)  ||  (model->config.norm_cfg[1] == NORM_RMSNORM)){

            size_t shape[] = {dim_stream};
            if (model->config.submodel_type == MODELTYPE_VISION_ENCODER) {
                sprintf(tensor_name, "%slayers.%zu.layer_norm2.weight", layer_prefix, layer);
            }
	    else{
                sprintf(tensor_name, "%slayers.%zu.post_attention_layernorm.weight", layer_prefix, layer);
            }
            add_tensor_metadata(tensor_name, shape, 1, &current_offset, json_buffer, dtype_str);
            
            if(model->config.norm_cfg[1] == NORM_LAYERNORM){
                if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
                    sprintf(tensor_name, "%slayers.%zu.layer_norm2.bias", layer_prefix, layer);
                }
		else{
                    sprintf(tensor_name, "%slayers.%zu.post_attention_layernorm.bias", layer_prefix, layer);
                }
                add_tensor_metadata(tensor_name, shape, 1, &current_offset, json_buffer, dtype_str);
            }
        }
        
        //FFN weights
        size_t shape_ffn1[] = {dim_stream, hidden_dim};
        if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
            sprintf(tensor_name, "%slayers.%zu.mlp.fc1.weight", layer_prefix, layer);
        }
	else{
            sprintf(tensor_name, "%slayers.%zu.mlp.gate_proj.weight", layer_prefix, layer);
        }
        add_tensor_metadata(tensor_name, shape_ffn1, 2, &current_offset, json_buffer, dtype_str);
        
        if(model->config.bias_cfg[0] == BIAS_ON){

            size_t shape_ffn1b[] = {hidden_dim};
            if (model->config.submodel_type == MODELTYPE_VISION_ENCODER) {
                sprintf(tensor_name, "%slayers.%zu.mlp.fc1.bias", layer_prefix, layer);
            }
	    else{
                sprintf(tensor_name, "%slayers.%zu.mlp.gate_proj.bias", layer_prefix, layer);
            }
            add_tensor_metadata(tensor_name, shape_ffn1b, 1, &current_offset, json_buffer, dtype_str);
        }
        
        //gate projection for LLAMA-style models
        if((model->config.ffn_nl_type[1] == GATE_ON) || (model->config.ffn_nl_type[0] == NL_SILU_LLAMA)){
            sprintf(tensor_name, "%slayers.%zu.mlp.up_proj.weight", layer_prefix, layer);
            add_tensor_metadata(tensor_name, shape_ffn1, 2, &current_offset, json_buffer, dtype_str);
        }
        
        size_t shape_ffn2[] = {hidden_dim, dim_stream};
        if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
            sprintf(tensor_name, "%slayers.%zu.mlp.fc2.weight", layer_prefix, layer);
        }
	else{
            sprintf(tensor_name, "%slayers.%zu.mlp.down_proj.weight", layer_prefix, layer);
        }
        add_tensor_metadata(tensor_name, shape_ffn2, 2, &current_offset, json_buffer, dtype_str);

        
        if(model->config.bias_cfg[1] == BIAS_ON){

            size_t shape_ffn2b[] = {dim_stream};

            if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
                sprintf(tensor_name, "%slayers.%zu.mlp.fc2.bias", layer_prefix, layer);
            }
	    else{
                sprintf(tensor_name, "%slayers.%zu.mlp.down_proj.bias", layer_prefix, layer);
            }
            add_tensor_metadata(tensor_name, shape_ffn2b, 1, &current_offset, json_buffer, dtype_str);
        }
    }
    

    //final normalization
    if((model->config.norm_cfg[2] == NORM_LAYERNORM)  ||  (model->config.norm_cfg[2] == NORM_RMSNORM)){

        size_t shape[] = {dim_stream};
        
        if((model->config.submodel_type == MODELTYPE_VISION_ENCODER)  &&  (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)){
            sprintf(tensor_name, "vision_tower.vision_model.post_layernorm.weight");
        }
	else{
            sprintf(tensor_name, "%snorm.weight", prefix);
        }
        add_tensor_metadata(tensor_name, shape, 1, &current_offset, json_buffer, dtype_str);
        

        if(model->config.norm_cfg[2] == NORM_LAYERNORM){

            if((model->config.submodel_type == MODELTYPE_VISION_ENCODER)  &&  (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)){
                sprintf(tensor_name, "vision_tower.vision_model.post_layernorm.bias");
            }
	    else{
                sprintf(tensor_name, "%snorm.bias", prefix);
            }
            add_tensor_metadata(tensor_name, shape, 1, &current_offset, json_buffer, dtype_str);
        }
    }
   


 

    //close JSON
    strcat(json_buffer, "}");
    
    //write header
    json_length = strlen(json_buffer);
    //space padding to the end
    while((json_length%8)!=0){
	strcat(json_buffer," ");
	json_length++;
    }
    fwrite(&json_length, sizeof(uint64_t), 1, f);
    fwrite(json_buffer, 1, json_length, f);
    
    //now write all tensor data in the same order
    current_offset = 0;
    
    //write vision encoder specific tensors
    if((model->config.submodel_type == MODELTYPE_VISION_ENCODER)  &&  (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)){
        
        size_t patch_elements = 3 * model->config.vision_patch_size * model->config.vision_patch_size * dim_stream;
        convert_and_write_tensor(f, model->w.vision_embeddings_w, patch_elements);
        
        convert_and_write_tensor(f, model->w.vision_embeddings_b, dim_stream);
        
        convert_and_write_tensor(f, model->w.learned_pose_w, model->config.vision_image_tokens * dim_stream);
        
        convert_and_write_tensor(f, model->w.multimodal_projector_w, dim_stream * model->config.target_dim_stream);
        
        convert_and_write_tensor(f, model->w.multimodal_projector_b, model->config.target_dim_stream);
    }
    
    //write decoder specific tensors
    if(model->config.submodel_type == MODELTYPE_DECODER){

        convert_and_write_tensor(f, model->w.embeddings, vocab_size * dim_stream);
        
        if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){
            convert_and_write_tensor(f, model->w.logits_classifier, vocab_size * dim_stream);
        }
    }
    
    //write per-layer tensors
    for(size_t layer = 0; layer < n_layers; layer++){

        //pre-norm
        if((model->config.norm_cfg[0] == NORM_LAYERNORM)  ||  (model->config.norm_cfg[0] == NORM_RMSNORM)){

            convert_and_write_tensor(f, model->w.norm_pre_w[layer], dim_stream);
            if(model->config.norm_cfg[0] == NORM_LAYERNORM){
                convert_and_write_tensor(f, model->w.norm_pre_b[layer], dim_stream);
            }
        }
        
        //attention weights

	//in safetensors format, query and key proj matrixes require "interleaving" when POSE are not LEARNED:
	//1) if the original model is already SAFETENSORS, weights are already interleaved in RAM, so no interleave activity is required now when saving
	//2) if PosE are learned, no interleaving is required as well

	bool need_interleave = ((checkpoint_type != CP_SAFETENSORS)  &&  (model->config.pose_cfg != POSE_LEARNED));
	
        if(need_interleave)	convert_and_write_tensor_interleaved(f, model->w.qm[layer], dim_stream * n_queries * dim_qkv, dim_stream);
	else			convert_and_write_tensor(f, model->w.qm[layer], dim_stream * n_queries * dim_qkv);

        if(need_interleave)	convert_and_write_tensor_interleaved(f, model->w.km[layer], dim_stream * n_keys * dim_qkv, dim_stream);
	else			convert_and_write_tensor(f, model->w.km[layer], dim_stream * n_keys * dim_qkv);

        convert_and_write_tensor(f, model->w.vm[layer], dim_stream * n_keys * dim_qkv);
        convert_and_write_tensor(f, model->w.om[layer], n_queries * dim_qkv * dim_stream);
        
        

        //attention biases
        if(model->config.bias_cfg[3] == BIAS_ON){
            convert_and_write_tensor(f, model->w.qb[layer], n_queries * dim_qkv);
            convert_and_write_tensor(f, model->w.kb[layer], n_keys * dim_qkv);
            convert_and_write_tensor(f, model->w.vb[layer], n_keys * dim_qkv);
            convert_and_write_tensor(f, model->w.ob[layer], dim_stream);
        }
        
        //post-norm
        if((model->config.norm_cfg[1] == NORM_LAYERNORM)  ||  (model->config.norm_cfg[1] == NORM_RMSNORM)){
            convert_and_write_tensor(f, model->w.norm_post_w[layer], dim_stream);
            if (model->config.norm_cfg[1] == NORM_LAYERNORM) {
                convert_and_write_tensor(f, model->w.norm_post_b[layer], dim_stream);
            }
        }
        
        //FFN
        convert_and_write_tensor(f, model->w.pre_ffn_w[layer], dim_stream * hidden_dim);
        if(model->config.bias_cfg[0] == BIAS_ON){
            convert_and_write_tensor(f, model->w.pre_ffn_b[layer], hidden_dim);
        }
        
        if((model->config.ffn_nl_type[1] == GATE_ON) || (model->config.ffn_nl_type[0] == NL_SILU_LLAMA)){
            convert_and_write_tensor(f, model->w.pre_ffn_w2[layer], dim_stream * hidden_dim);
        }
        
        convert_and_write_tensor(f, model->w.post_ffn_w[layer], hidden_dim * dim_stream);
        if(model->config.bias_cfg[1] == BIAS_ON){
            convert_and_write_tensor(f, model->w.post_ffn_b[layer], dim_stream);
        }
    }
    
    //final norm
    if((model->config.norm_cfg[2] == NORM_LAYERNORM)  ||  (model->config.norm_cfg[2] == NORM_RMSNORM)){
        convert_and_write_tensor(f, model->w.norm_final_w, dim_stream);
        if(model->config.norm_cfg[2] == NORM_LAYERNORM){
            convert_and_write_tensor(f, model->w.norm_final_b, dim_stream);
        }
    }
    
    //clean up
    free(json_buffer);
    fclose(f);
    
    mylog(LOG_INFO, "Model saved successfully to file %s",filename);
    return 0;
}

void save_loss(float * sequence_loss, int * n_intoks, size_t B){

    FILE * f;
    time_t rawtime;
    struct tm * timeinfo;
    char time_str[32];
    char model_name[256];
    
    //extract model name from mod_path (global variable)
    //get just the filename without path
    char * base_name = strrchr(mod_path, '/');
    if(base_name){
        base_name++; //skip the '/'
    }
    else{
        base_name = mod_path;
    }
   

    strcpy(model_name, base_name); 

    
    //get current timestamp
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    //open file in append mode
    f = fopen("loss.csv", "a");
    if(f == NULL){
        mylog(LOG_ERROR, "Failed to open loss.csv for appending");
        return;
    }
    
    //check if file is empty (new file) and add header if needed
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if(file_size == 0){
        //write header
        fprintf(f, "checkpoint;timestamp;step;learning_rate;batch_aveloss;");
        for(size_t b = 0; b < B; b++){
            fprintf(f, ";loss#%zu", b);
        }
        fprintf(f, "\n");
    }
    
    //write data row
    fprintf(f, "%s;%s;%05zu;%.9f", model_name, time_str, adamw_cfg.step,adamw_cfg.learning_rate);
   
    //calculate and write average loss in batch
    float batch_aveloss = 0.0;
    for(size_t b = 0; b < B; b++){
        float mean_loss = sequence_loss[b] / ((float)n_intoks[b]);
	batch_aveloss += mean_loss;
    }
    batch_aveloss /= (float)B;
    fprintf(f, ";%.9f;", batch_aveloss);
 
    //write mean losses for each sequence
    for(size_t b = 0; b < B; b++){
        float mean_loss = sequence_loss[b] / ((float)n_intoks[b]);
        fprintf(f, ";%.3f", mean_loss);
    }
    fprintf(f, ";\n");
    
    fclose(f);
    
    mylog(LOG_VERBOSE_DEBUG, "Losses saved to loss.csv");
}




// ============================================================
//  Model configuration and initialization
// ============================================================

void check_model_configuration(Model * model){

	if(model->config.n_queries <= 0){
		mylog(LOG_ERROR, "Bad model configuration: n_queries = %d", model->config.n_queries);
		exit(-1);
	}


	if((model->config.dim_stream % 2) != 0){
		mylog(LOG_ERROR, "Bad model configuration: dim_stream (%d) must be EVEN to properly support positional embeddings", model->config.dim_stream);
		exit(-1);
	}


	if((action == ACTION_CHAT) && (chat_scheme == CHATSCHEME_NONE)){
		mylog(LOG_ERROR, "Bad model configuration: chat mode requested, but no chat scheme specified, and architecture \"%s\" has no default chat scheme",arch_text[model->config.architectures]);
		exit(-1);
	}
}



void print_model_configuration(Model * model){

	mylog(LOG_INFO, "DATA TYPE: %s %s", wtype_text[wtype], ((wtype==WTYPE_UNDEFINED)?"(yet)":""));
	mylog(LOG_INFO, "ARCHITECTURE: %s", arch_text[model->config.architectures]);
	mylog(LOG_INFO, "dim_stream = %d", model->config.dim_stream);
	mylog(LOG_INFO, "ffn_hidden_dim = %d", model->config.ffn_hidden_dim);
	mylog(LOG_INFO, "n_layers = %d", model->config.n_layers);
	mylog(LOG_INFO, "n_queries = %d", model->config.n_queries);
	mylog(LOG_INFO, "n_keys = %d", model->config.n_keys);

	if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){
		
		mylog(LOG_INFO, "vision_image_tokens = %d", model->config.vision_image_tokens);
		mylog(LOG_INFO, "vision_patch_size = %d", model->config.vision_patch_size);
		mylog(LOG_INFO, "target_dim_stream = %d", model->config.target_dim_stream);
	}
	else{
		mylog(LOG_INFO, "vocab_size = %d", model->config.vocab_size);
		mylog(LOG_INFO, "sequence_maxtokens = %d", model->config.sequence_maxtokens);
		mylog(LOG_INFO, "embeddings: %s", embeddings_cfg_text[model->config.embeddings_cfg]);
	}

	mylog(LOG_INFO, "positional embeddings: %s", pose_cfg_text[model->config.pose_cfg]);

	mylog(LOG_INFO, "layer normalization pre-attention: %s", norm_cfg_text[model->config.norm_cfg[0]]);
	mylog(LOG_INFO, "layer normalization post-attention: %s", norm_cfg_text[model->config.norm_cfg[1]]);
	mylog(LOG_INFO, "layer normalization final: %s", norm_cfg_text[model->config.norm_cfg[2]]);

	mylog(LOG_INFO, "non-linearity type: %s", ffn_nl_type_text[model->config.ffn_nl_type[0]]);
	mylog(LOG_INFO, "biases before non-linearity: %s", bias_cfg_text[model->config.bias_cfg[0]]);
	mylog(LOG_INFO, "biases after non-linearity: %s", bias_cfg_text[model->config.bias_cfg[1]]);

	mylog(LOG_INFO, "norm epsilon: %f", norm_eps);

	if(model->config.pose_cfg != POSE_LEARNED){
		mylog(LOG_INFO, "positional embeddings theta: %f", pose_theta);
	}
	if(action == ACTION_CHAT){
		mylog(LOG_INFO, "chatscheme: %s", chatscheme_text[chat_scheme]);
	}
}



ssize_t init_model(Model * model, int checkpoint_type, Model * brother_model){


	byte * p;			//current offset in checkpoint
	ssize_t s;			//step to next element


	
	if(	((checkpoint_type == CP_LLAMA2_AK)  ||  (checkpoint_type == CP_GPT2_AK))
		&&
		(model->file_data != NULL)	//if a checkpoint was loaded
	){


		//LOAD CONFIGURATION
		if(checkpoint_type == CP_LLAMA2_AK){	//checkpoint file for LLAMA2 C implementation from A.Karpathy
							
			p = model->file_data[0] + 0;		//we have just one file, we put the pointer at the beginning of it
			s = sizeof(int);

			memcpy(&model->config.dim_stream,         p, s);		p += s;
			memcpy(&model->config.ffn_hidden_dim,     p, s);		p += s;
			memcpy(&model->config.n_layers,           p, s);		p += s;
			memcpy(&model->config.n_queries,          p, s);		p += s;
			memcpy(&model->config.n_keys,             p, s);		p += s;
			memcpy(&model->config.vocab_size,         p, s);		p += s;
			memcpy(&model->config.sequence_maxtokens, p, s);		p += s;

			if(model->config.vocab_size < 0){	//strange way to signal "unshared embeddings"
				model->config.vocab_size *= -1;	
				model->config.embeddings_cfg = EMBEDDINGS_UNSHARED;
			}
			else{
				model->config.embeddings_cfg = EMBEDDINGS_SHARED;
			}

			model->config.pose_cfg = POSE_ROPE;

			model->config.norm_cfg[0] = NORM_RMSNORM; 
			model->config.norm_cfg[1] = NORM_RMSNORM; 
			model->config.norm_cfg[2] = NORM_RMSNORM;

			model->config.ffn_nl_type[0] = NL_SILU_LLAMA;
			model->config.ffn_nl_type[1] = GATE_ON;

			model->config.bias_cfg[0] = BIAS_OFF;
			model->config.bias_cfg[1] = BIAS_OFF;


			wtype = WTYPE_FLOAT32;
			wsize = 4;

			model->config.architectures = ARCH_LLAMA_CAUSAL;
			model->config.submodel_type = MODELTYPE_DECODER; 
		}
		else
		if(checkpoint_type == CP_GPT2_AK){	//checkpoint file for GPT2 C implementation from A.Karpathy
							
			p = model->file_data[0] + 0;		//we have just one file, we put the pointer at the beginning of it
			s = sizeof(int);

			p += s;	//skip magic number
			p += s;	//skip version number

			memcpy(&model->config.sequence_maxtokens, p, s);		p += s;
			memcpy(&model->config.vocab_size,         p, s);		p += s;	//this could be overwritten 4 rows below...
			memcpy(&model->config.n_layers,           p, s);		p += s;
			memcpy(&model->config.n_queries,          p, s);		p += s;
			memcpy(&model->config.dim_stream,         p, s);		p += s;
			memcpy(&model->config.vocab_size,         p, s);		p += s;	//padded vocabulary size, we want this one

			model->config.ffn_hidden_dim = model->config.dim_stream * 4;	//GPT2-specific

			/* memcpy(&model->config.n_keys,             p, s); */		p += s;
			model->config.n_keys = model->config.n_queries;			//GPT-2 has multi-head attention

			model->config.embeddings_cfg = EMBEDDINGS_SHARED;		//GPT-2 has shared embeddings
			

			model->config.pose_cfg = POSE_LEARNED;

			model->config.norm_cfg[0] = NORM_LAYERNORM; 
			model->config.norm_cfg[1] = NORM_LAYERNORM; 
			model->config.norm_cfg[2] = NORM_LAYERNORM;

			model->config.ffn_nl_type[0] = NL_GELU_TANH;
			model->config.ffn_nl_type[1] = GATE_OFF;

			model->config.bias_cfg[0] = BIAS_ON;	//linear layer before FFN has bias in GPT-2
			model->config.bias_cfg[1] = BIAS_ON;	//linear layer after  FFN has bias in GPT-2
			//model->config.bias_cfg[2] = BIAS_ON;	//not used
			model->config.bias_cfg[3] = BIAS_ON;	//attention projections have bias in GPT-2


			wtype = WTYPE_FLOAT32;
			wsize = 4;

			model->config.architectures = ARCH_GPT2_CAUSAL;
			model->config.submodel_type = MODELTYPE_DECODER; 
		}
	}
	else
	if(checkpoint_type == CP_SAFETENSORS){	//checkpoint file in SafeTensors format: there's an extra configuration file
		
		FILE * f;
		size_t size;
		int alloc_flag = 0;

		if(cfg_path == NULL){
			model->dirpath = calloc(1024, sizeof(char));	//+1 is just to handle terminator
			cfg_path = calloc(strlen(model->dirpath)+16 , sizeof(char));

			if(strlen(model->dirpath) > 0)	sprintf(cfg_path, "%s/config.json", model->dirpath);
			else				sprintf(cfg_path, "config.json");
			alloc_flag = 1;
		}

		if((f=fopen(cfg_path,"rb")) == NULL){
			mylog(LOG_ERROR,"Cannot open the configuration file \"%s\". Exiting...",cfg_path); exit(-1);
		}

		fseek(f,0,SEEK_END);
		size = ftell(f);
		fclose(f);

		if(size<0){
			mylog(LOG_ERROR,"Configuration file \"%s\" size is invalid (<0). Exiting...",cfg_path); exit(-1); 
		}

		char * json;
		json = calloc(size+1,sizeof(char));	//+1 for terminator

		if((f=fopen(cfg_path,"rb")) == NULL){
			mylog(LOG_ERROR,"Cannot reopen the configuration file \"%s\". Exiting...",cfg_path); exit(-1);
		}

		if(fread(json,sizeof(char),size,f) != size){
			fclose(f);
			mylog(LOG_ERROR,"Cannot read the configuration file \"%s\". Exiting...",cfg_path); exit(-1);
		}
	
		fclose(f);

		if(alloc_flag==1) {free(cfg_path); cfg_path=NULL;}


		char * json_pointer = json;	
		JsonNode * cfg = parseJsonValue(&json_pointer);
		if(log_cfg >= LOG_DEBUG){
			printJsonTree(cfg, 0);
		}
		free(json);


		//Now, let's translate from json tree to configuration 
		
		JsonNode * jn;
		int layer;

		layer = -1;

		jn = json_get(cfg, "configuration", "architectures",      TRIPTYPE_INT,    &model->config.architectures,         layer, "basic");



		if( (model->config.architectures == ARCH_LLAMA_CAUSAL)
		    ||
		    (model->config.architectures == ARCH_GEMMA_CAUSAL)
		){


			jn = json_get(cfg, "configuration", "vocab_size",         TRIPTYPE_INT,    &model->config.vocab_size,            layer, NULL);
			jn = json_get(cfg, "configuration", "sequence_maxtokens", TRIPTYPE_INT,    &model->config.sequence_maxtokens,    layer, NULL);
			jn = json_get(cfg, "configuration", "embeddings_cfg",     TRIPTYPE_INT,    &model->config.embeddings_cfg,        layer, "true"); //it's a bool, not a string! Would be "\"true\"" to be a string


			jn = json_get(cfg, "configuration", "dim_stream",         TRIPTYPE_INT,    &model->config.dim_stream,            layer, NULL);
			jn = json_get(cfg, "configuration", "ffn_hidden_dim",     TRIPTYPE_INT,    &model->config.ffn_hidden_dim,        layer, NULL);
			jn = json_get(cfg, "configuration", "n_layers",           TRIPTYPE_INT,    &model->config.n_layers,              layer, NULL);
			jn = json_get(cfg, "configuration", "n_queries",          TRIPTYPE_INT,    &model->config.n_queries,             layer, NULL);
			jn = json_get(cfg, "configuration", "n_keys",             TRIPTYPE_INT,    &model->config.n_keys,                layer, NULL);
			jn = json_get(cfg, "configuration", "pose_cfg",           TRIPTYPE_INT,    &model->config.pose_cfg,              layer, int2str(POSE_ROPE));
			jn = json_get(cfg, "configuration", "norm_cfg[0]",        TRIPTYPE_INT,    &model->config.norm_cfg[0],           layer, int2str(NORM_RMSNORM));
			jn = json_get(cfg, "configuration", "norm_cfg[1]",        TRIPTYPE_INT,    &model->config.norm_cfg[1],           layer, int2str(model->config.norm_cfg[0]));
			jn = json_get(cfg, "configuration", "norm_cfg[2]",        TRIPTYPE_INT,    &model->config.norm_cfg[2],           layer, int2str(model->config.norm_cfg[0]));

			jn = json_get(cfg, "configuration", "ffn_nl_type[0]",     TRIPTYPE_INT,    &model->config.ffn_nl_type[0],        layer, NULL);
			jn = json_get(cfg, "configuration", "ffn_nl_type[1]",     TRIPTYPE_INT,    &model->config.ffn_nl_type[1],        layer, int2str((model->config.ffn_nl_type[0]==NL_SILU_LLAMA) ? GATE_ON : (model->config.architectures == ARCH_GEMMA_CAUSAL) ? GATE_ON : GATE_OFF));

			jn = json_get(cfg, "configuration", "bias_cfg[0]",        TRIPTYPE_INT,    &model->config.bias_cfg[0],           layer, int2str(BIAS_OFF));
			jn = json_get(cfg, "configuration", "bias_cfg[1]",        TRIPTYPE_INT,    &model->config.bias_cfg[1],           layer, int2str(model->config.bias_cfg[0]));
			jn = json_get(cfg, "configuration", "bias_cfg[2]",        TRIPTYPE_INT,    &model->config.bias_cfg[2],           layer, int2str(model->config.bias_cfg[0]));
			jn = json_get(cfg, "configuration", "bias_cfg[3]",        TRIPTYPE_INT,    &model->config.bias_cfg[3],           layer, int2str(model->config.bias_cfg[0]));

			jn = json_get(cfg, "configuration", "norm_eps",           TRIPTYPE_FLOAT,  &norm_eps,                            layer, float2str(norm_eps));
			jn = json_get(cfg, "configuration", "pose_theta",         TRIPTYPE_FLOAT,  &pose_theta,                          layer, float2str(pose_theta));

		}
		else
		if(model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL){

			JsonNode * vision_cfg;
			JsonNode * lmodel_cfg;

			vision_cfg = findJsonNodeByKey(cfg, "vision_config"); 
			lmodel_cfg = findJsonNodeByKey(cfg, "text_config");


			if(vision_cfg == NULL){
				mylog(LOG_ERROR, "Vision model configuration not found in configuration file \"%s\"!",cfg_path);
			}
			if(lmodel_cfg == NULL){
				mylog(LOG_ERROR, "Language model configuration not found in configuration file \"%s\"!",cfg_path);
			}
			if( (vision_cfg == NULL)  ||  (lmodel_cfg == NULL) ){
				mylog(LOG_ERROR, "Exiting...");
				exit(1);
			}


			if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){

				char * file_text = "vm_configuration";

				//vision model extra configuration (encoder)
				jn = json_get(vision_cfg, file_text, "vision_image_tokens",     TRIPTYPE_INT,    &model->config.vision_image_tokens,	-1, NULL);
				jn = json_get(vision_cfg, file_text, "vision_patch_size", 	TRIPTYPE_INT,    &model->config.vision_patch_size,	-1, NULL);
				jn = json_get(vision_cfg, file_text, "target_dim_stream",	TRIPTYPE_INT,    &model->config.target_dim_stream,	-1, NULL);


				//vision model configuration (encoder)
				jn = json_get(vision_cfg, file_text, "dim_stream",         TRIPTYPE_INT,    &model->config.dim_stream,         -1, NULL);
				jn = json_get(vision_cfg, file_text, "ffn_hidden_dim",     TRIPTYPE_INT,    &model->config.ffn_hidden_dim,     -1, NULL);
				jn = json_get(vision_cfg, file_text, "n_layers",           TRIPTYPE_INT,    &model->config.n_layers,           -1, NULL);
				jn = json_get(vision_cfg, file_text, "n_queries",          TRIPTYPE_INT,    &model->config.n_queries,          -1, NULL);
				jn = json_get(vision_cfg, file_text, "n_keys",             TRIPTYPE_INT,    &model->config.n_keys,             -1, int2str(model->config.n_queries));
				jn = json_get(vision_cfg, file_text, "pose_cfg",           TRIPTYPE_INT,    &model->config.pose_cfg,           -1, int2str(POSE_LEARNED));
				jn = json_get(vision_cfg, file_text, "norm_cfg[0]",        TRIPTYPE_INT,    &model->config.norm_cfg[0],        -1, int2str(NORM_LAYERNORM));
				jn = json_get(vision_cfg, file_text, "norm_cfg[1]",        TRIPTYPE_INT,    &model->config.norm_cfg[1],        -1, int2str(model->config.norm_cfg[0]));
				jn = json_get(vision_cfg, file_text, "norm_cfg[2]",        TRIPTYPE_INT,    &model->config.norm_cfg[2],        -1, int2str(model->config.norm_cfg[0]));

				jn = json_get(vision_cfg, file_text, "ffn_nl_type[0]",     TRIPTYPE_INT,    &model->config.ffn_nl_type[0],     -1, int2str(NL_GELU_TANH));
				jn = json_get(vision_cfg, file_text, "ffn_nl_type[1]",     TRIPTYPE_INT,    &model->config.ffn_nl_type[1],     -1, int2str((model->config.ffn_nl_type[0]==NL_SILU_LLAMA) ? GATE_ON : (model->config.architectures == ARCH_GEMMA_CAUSAL) ? GATE_ON : GATE_OFF));

				jn = json_get(vision_cfg, file_text, "bias_cfg[0]",        TRIPTYPE_INT,    &model->config.bias_cfg[0],        -1, int2str(BIAS_ON));
				jn = json_get(vision_cfg, file_text, "bias_cfg[1]",        TRIPTYPE_INT,    &model->config.bias_cfg[1],        -1, int2str(model->config.bias_cfg[0]));
				jn = json_get(vision_cfg, file_text, "bias_cfg[2]",        TRIPTYPE_INT,    &model->config.bias_cfg[2],        -1, int2str(BIAS_OFF));
				jn = json_get(vision_cfg, file_text, "bias_cfg[3]",        TRIPTYPE_INT,    &model->config.bias_cfg[3],        -1, int2str(BIAS_ON));

				jn = json_get(vision_cfg, file_text, "norm_eps",           TRIPTYPE_FLOAT,  &norm_eps,                         -1, float2str(1e-06));
				//jn = json_get(vision_cfg, file_text, "pose_theta",         TRIPTYPE_FLOAT,  &pose_theta,                       -1, float2str(pose_theta));

			}
			else
			if(model->config.submodel_type == MODELTYPE_DECODER){

				char * file_text = "lm_configuration";


				//language model extra configuration (decoder)
				jn = json_get(lmodel_cfg, file_text, "vocab_size",         TRIPTYPE_INT,    &model->config.vocab_size,         -1, NULL);
				jn = json_get(lmodel_cfg, file_text, "sequence_maxtokens", TRIPTYPE_INT,    &model->config.sequence_maxtokens, -1, int2str(8192));
				jn = json_get(lmodel_cfg, file_text, "embeddings_cfg",     TRIPTYPE_INT,    &model->config.embeddings_cfg,     -1, "true"); //it's a bool, not a string! Would be "\"true\"" to be a string

				//language model configuration (decoder)
				jn = json_get(lmodel_cfg, file_text, "dim_stream",         TRIPTYPE_INT,    &model->config.dim_stream,         -1, NULL);
				jn = json_get(lmodel_cfg, file_text, "ffn_hidden_dim",     TRIPTYPE_INT,    &model->config.ffn_hidden_dim,     -1, NULL);
				jn = json_get(lmodel_cfg, file_text, "n_layers",           TRIPTYPE_INT,    &model->config.n_layers,           -1, NULL);
				jn = json_get(lmodel_cfg, file_text, "n_queries",          TRIPTYPE_INT,    &model->config.n_queries,          -1, NULL);
				jn = json_get(lmodel_cfg, file_text, "n_keys",             TRIPTYPE_INT,    &model->config.n_keys,             -1, NULL);
				jn = json_get(lmodel_cfg, file_text, "pose_cfg",           TRIPTYPE_INT,    &model->config.pose_cfg,           -1, int2str(POSE_ROPE));
				jn = json_get(lmodel_cfg, file_text, "norm_cfg[0]",        TRIPTYPE_INT,    &model->config.norm_cfg[0],        -1, int2str(NORM_RMSNORM));
				jn = json_get(lmodel_cfg, file_text, "norm_cfg[1]",        TRIPTYPE_INT,    &model->config.norm_cfg[1],        -1, int2str(model->config.norm_cfg[0]));
				jn = json_get(lmodel_cfg, file_text, "norm_cfg[2]",        TRIPTYPE_INT,    &model->config.norm_cfg[2],        -1, int2str(model->config.norm_cfg[0]));

				jn = json_get(lmodel_cfg, file_text, "ffn_nl_type[0]",     TRIPTYPE_INT,    &model->config.ffn_nl_type[0],     -1, int2str(NL_GELU_TANH));
				jn = json_get(lmodel_cfg, file_text, "ffn_nl_type[1]",     TRIPTYPE_INT,    &model->config.ffn_nl_type[1],     -1, int2str((model->config.ffn_nl_type[0]==NL_SILU_LLAMA) ? GATE_ON : (model->config.architectures == ARCH_GEMMA_CAUSAL) ? GATE_ON : (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL) ? GATE_ON : GATE_OFF));

				jn = json_get(lmodel_cfg, file_text, "bias_cfg[0]",        TRIPTYPE_INT,    &model->config.bias_cfg[0],        -1, int2str(BIAS_OFF));
				jn = json_get(lmodel_cfg, file_text, "bias_cfg[1]",        TRIPTYPE_INT,    &model->config.bias_cfg[1],        -1, int2str(model->config.bias_cfg[0]));
				jn = json_get(lmodel_cfg, file_text, "bias_cfg[2]",        TRIPTYPE_INT,    &model->config.bias_cfg[2],        -1, int2str(model->config.bias_cfg[0]));
				jn = json_get(lmodel_cfg, file_text, "bias_cfg[3]",        TRIPTYPE_INT,    &model->config.bias_cfg[3],        -1, int2str(model->config.bias_cfg[0]));

				jn = json_get(lmodel_cfg, file_text, "norm_eps",           TRIPTYPE_FLOAT,  &norm_eps,                         -1, float2str(norm_eps));
				jn = json_get(lmodel_cfg, file_text, "pose_theta",         TRIPTYPE_FLOAT,  &pose_theta,                       -1, float2str(pose_theta));

			}





		}
		else{
			mylog(LOG_ERROR, "Architecture \"%s\" configuration not (yet) handled for safetensors format in init_model(). Exiting...", arch_text[model->config.architectures]);
			exit(1);
		}

		freeJsonTree(cfg);
		cfg = NULL;
	}

	




	

	//if(action == ACTION_CHAT){
	  if(chat_scheme == CHATSCHEME_NONE){
		if(model->config.architectures == ARCH_LLAMA_CAUSAL)	chat_scheme = CHATSCHEME_LLAMA;
		if(model->config.architectures == ARCH_GEMMA_CAUSAL)	chat_scheme = CHATSCHEME_GEMMA;
	  }
	//}



	print_model_configuration(model);

	check_model_configuration(model);



	if(brother_model != NULL){
		model->config.shared_weights_memory = true;
	}
	else{
		model->config.shared_weights_memory = false;
	}




	if(action != ACTION_CREATE){

		if(ramflag == RAM_NO_OPTIMIZATIONS){

		    if(brother_model != NULL){

			for(int i=0; i < model->nfiles; i++){

				//we can immediately memory-unmap the checkpoint file, and close it
				munmap(model->file_data[i], model->checkpoint_size[i]);
				close(model->fd[i]);
	
				model->tensors_data[i] = brother_model->tensors_data[i];
				model->header_data[i]  = brother_model->header_data[i];

			}

		    }
		    else{

			size_t header_size;
			size_t weights_size;
			byte * weights_memory = NULL;
			byte * header_memory = NULL;



			for(int i=0; i < model->nfiles; i++){
				header_size  = get_checkpointfile_headersize(model, i);	//if SAFETENSORS, it changes from file to file
				weights_size = model->checkpoint_size[i] - header_size;
	
				//we allocate ACTUAL RAM space for the weights; we want them nicely aligned to the L1 cache line size, see matmulf
				//posix_memalign( (byte **)&weights_memory, CACHE_LINESIZE, weights_size);
				weights_memory = myalloc( weights_size );

				if((weights_memory == NULL) || (errno==ENOMEM)){
					mylog(LOG_ERROR,"Cannot allocate enough RAM space for checkpoint file #%d. Try using \"--ram\". Exiting...",i); exit(-1);
				}
	
//printf("\n model->file_data[%d] was = %16X \t model->checkpoint_size = %zd \n", i, model->file_data[i], model->checkpoint_size[i]);
				//this memcpy will actually read all the weights from the checkpoint file (currently memory-mapped) to the RAM
				memcpy(weights_memory, model->file_data[i] + header_size, weights_size);


				if(checkpoint_type == CP_SAFETENSORS){	//checkpoint file in SafeTensors format: there's an extra configuration file

					header_memory = myalloc( header_size );
					memcpy(header_memory, model->file_data[i], header_size);
				}

				//now we can memory-unmap the checkpoint file, and close it
				munmap(model->file_data[i], model->checkpoint_size[i]);
				close(model->fd[i]);
	
				model->tensors_data[i] = (unsigned char *)weights_memory;
				model->header_data[i]  = (unsigned char *)header_memory;

//printf("\n model->header_data[%d] is = %16X \t header_size = %zd \n", i, model->header_data[i], header_size);
//printf("\n model->tensors_data[%d] is = %16X \t weights_size = %zd \n", i, model->tensors_data[i], weights_size);

				//model->checkpoint_size[i] = header_size + weights_size;	//unchanged
			}
		    }
		
		
		}
		else{   //if RAM OPIMIZATIONS are active 
		
		    if(brother_model != NULL){

			for(int i=0; i < model->nfiles; i++){

				model->tensors_data[i] = brother_model->tensors_data[i];
				model->header_data[i]  = brother_model->header_data[i];

			}

		    }
		    else{


			size_t header_size;
			size_t weights_size;


			for(int i=0; i < model->nfiles; i++){

				header_size  = get_checkpointfile_headersize(model, i);	//if SAFETENSORS, it changes from file to file
				weights_size = model->checkpoint_size[i] - header_size;
	
//printf("\n model->file_data[%d] is = %16X \t model->checkpoint_size = %zd \n", i, model->file_data[i], model->checkpoint_size[i]);

				model->header_data[i]  = (model->file_data[i] + 0);
				model->tensors_data[i] = (model->file_data[i] + header_size);

//printf("\n model->heeeeader_data[%d] is = %16X \t header_size = %zd \n", i, model->header_data[i], header_size);
//printf("\n model->tensors_data[%d] is = %16X \t weights_size = %zd \n", i, model->tensors_data[i], weights_size);

				//model->checkpoint_size[i] = header_size + weights_size;	//unchanged

			}
		    }

		}
	




		//LOAD WEIGHTS (weights are already loaded, we just need to map the pointers to them in the RAM)


		//let's define some variable for fast reference
		size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
		size_t dim_stream = (size_t)(model->config.dim_stream);
		//size_t max_tokens = (size_t)(model->config.sequence_maxtokens);
		size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
		size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
		size_t vocab_size = (size_t)(model->config.vocab_size);
		size_t n_layers   = (size_t)(model->config.n_layers);
		size_t n_queries  = (size_t)(model->config.n_queries);
		size_t n_keys     = (size_t)(model->config.n_keys);




		//let's allocate the space for the POINTERS for all the PER-LAYER tensors
		model->w.norm_pre_w = myalloc(n_layers * sizeof(byte *));
		model->w.norm_pre_b = myalloc(n_layers * sizeof(byte *));
		model->w.qm = myalloc(n_layers * sizeof(byte *));
		model->w.km = myalloc(n_layers * sizeof(byte *));
		model->w.vm = myalloc(n_layers * sizeof(byte *));
		model->w.om = myalloc(n_layers * sizeof(byte *));
		if(model->config.bias_cfg[3] == BIAS_ON){
			model->w.qb = myalloc(n_layers * sizeof(byte *));
			model->w.kb = myalloc(n_layers * sizeof(byte *));
			model->w.vb = myalloc(n_layers * sizeof(byte *));
			model->w.ob = myalloc(n_layers * sizeof(byte *));
		}
		model->w.norm_post_w = myalloc(n_layers * sizeof(byte *));
		model->w.norm_post_b = myalloc(n_layers * sizeof(byte *));
		model->w.pre_ffn_w = myalloc(n_layers * sizeof(byte *));
		model->w.pre_ffn_b = myalloc(n_layers * sizeof(byte *));
		model->w.pre_ffn_w2 = myalloc(n_layers * sizeof(byte *));
		model->w.post_ffn_w = myalloc(n_layers * sizeof(byte *));
		model->w.post_ffn_b = myalloc(n_layers * sizeof(byte *));




		if(checkpoint_type == CP_LLAMA2_AK)		//checkpoint file for LLAMA2 C implementation from A.Karpathy
		{

			p = model->tensors_data[0] + 0;


			s = sizeof(float) * vocab_size * dim_stream;
			model->w.embeddings = p;
			p += s;
	
	
			//PRE-NORM: weights
			if( (model->config.norm_cfg[0] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[0] == NORM_RMSNORM)		//RMSNORM
			){
				s = sizeof(float) * dim_stream;

				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_pre_w[layer] = p;
					p += s;
				}
			}
	
	
			//PRE-NORM: biases
			if( (model->config.norm_cfg[0] == NORM_LAYERNORM)	//LAYERNORM
			){
				s = sizeof(float) * dim_stream;

				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_pre_b[layer] = p;
					p += s;
				}
			}
	
	
	
	
	
			//QUERY matrixes
			s = (sizeof(float) * dim_stream * dim_qkv) * n_queries;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.qm[layer] = p;
				p += s;
			}
	
			//KEY matrixes
			s = (sizeof(float) * dim_stream * dim_qkv) * n_keys;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.km[layer] = p;
				p += s;
			}
			
			//VALUE matrixes
			//size is the SAME as KEY matrixes: 
			s = (sizeof(float) * dim_stream * dim_qkv) * n_keys;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.vm[layer] = p;
				p += s;
			}
	
			//OUTPUT matrixes (one for each head/query)
			//size is the SAME as QUERY matrixes;    BUT:
			//1) they are (dim_qkv,dim_stream) instead of (dim_stream,dim_qkv)
			//2) their lines are interleaved, because we are going to do a single matmul to get the output projection from the concatenation of all the heads outputs
			s = (sizeof(float) * dim_stream * dim_qkv) * n_queries;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.om[layer] = p;
				p += s;
			}
	
	
	
	
			//POST-NORM: weights
			if( (model->config.norm_cfg[1] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[1] == NORM_RMSNORM)		//RMSNORM
			){
				s = sizeof(float) * dim_stream;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_post_w[layer] = p;
					p += s;
				}
			}
	
	
			//POST-NORM: biases
			if( (model->config.norm_cfg[1] == NORM_LAYERNORM)	//LAYERNORM
			){
				s = sizeof(float) * dim_stream;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_post_b[layer] = p;
					p += s;
				}
			}
	
	
	
			//LINEAR LAYER PRE-FeedForwardNetwork
	
			//weights
			s = sizeof(float) * dim_stream * hidden_dim;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.pre_ffn_w[layer] = p;
				p += s;
			}	
		
			//biases, if required
			if(model->config.bias_cfg[0] == BIAS_ON){
				s = sizeof(float) * hidden_dim;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.pre_ffn_b[layer] = p;
					p += s;
				}
			}
	

	
			//LINEAR LAYER POST-Feed Forward Network
	
			//weights
			s = sizeof(float) * dim_stream * hidden_dim;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.post_ffn_w[layer] = p;
				p += s;
			}
			
			//biases, if required
			if(model->config.bias_cfg[1] == BIAS_ON){
				s = sizeof(float) * dim_stream;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.post_ffn_b[layer] = p;
					p += s;
				}
			}


							
			//ONE STEP BACKWARD: INSIDE the Feed Forward Network itself:
			//no special weights are required here; all the mess is in the two linear layers (pre+post)
			//... UNLESS ...
			//special case: e.g.: LLAMA2 SILU non-linearity, which requires additional multipliers
			if(	(model->config.ffn_nl_type[1] == GATE_ON)
				||
				(model->config.ffn_nl_type[0] == NL_SILU_LLAMA)
			){
				s = sizeof(float) * dim_stream * hidden_dim;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.pre_ffn_w2[layer] = p;
					p += s;
				}
			}
	
	
	
			//FINAl-NORM: weights
			if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[2] == NORM_RMSNORM)		//RMSNORM
			){
				s = sizeof(float) * dim_stream;
				model->w.norm_final_w = p;
				p += s;
			}



			//for compatibility
			if(checkpoint_type == CP_LLAMA2_AK){	//checkpoint file for LLAMA2 C implementation from A.Karpathy
				//MUST SKIP RoPE coefficients!
				s = max_tokens * (dim_qkv/2);
				p += s;	//let's skip sin coefficients
				p += s; //let's skip cos coefficients 
			}



			//FINAL-NORM: biases
			if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
			){
				s = sizeof(float) * dim_stream;
				model->w.norm_final_b = p;
				p += s;
			}
	
	
	
			//FINAL LINEAR LAYER (classifier to logits)
	
			//weights
			if(model->config.embeddings_cfg == EMBEDDINGS_SHARED){
				model->w.logits_classifier = model->w.embeddings;
			}
			else
			if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){
				s = sizeof(float) * dim_stream * vocab_size;
				model->w.logits_classifier = p;
				p += s;
			}

		}
		else
		if(checkpoint_type == CP_GPT2_AK)	//checkpoint file for GPT2 C implementation from A.Karpathy
		{


			p = model->tensors_data[0] + 0;


			s = sizeof(float) * vocab_size * dim_stream;
			model->w.embeddings = p;
			p += s;


			s = sizeof(float) * max_tokens * dim_stream;
			model->w.learned_pose_w = p;
			p += s;




			//PRE-NORM: weights
			if( (model->config.norm_cfg[0] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[0] == NORM_RMSNORM)		//RMSNORM
			){
				s = sizeof(float) * dim_stream;

				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_pre_w[layer] = p;
					p += s;
				}
			}
	
	
			//PRE-NORM: biases
			if( (model->config.norm_cfg[0] == NORM_LAYERNORM)	//LAYERNORM
			){
				s = sizeof(float) * dim_stream;

				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_pre_b[layer] = p;
					p += s;
				}
			}
	
	
	
	
	
			//QUERY, KEY, VALUE matrixes (packed together)
			for(int layer = 0; layer < n_layers; layer++){

				s = (sizeof(float) * dim_stream * dim_qkv) * n_queries;
				model->w.qm[layer] = p;
				p += s;

				s = (sizeof(float) * dim_stream * dim_qkv) * n_keys;
				model->w.km[layer] = p;
				p += s;

				s = (sizeof(float) * dim_stream * dim_qkv) * n_keys;
				model->w.vm[layer] = p;
				p += s;
			}

			//QUERY, KEY, VALUE biases (packed together)
			for(int layer = 0; layer < n_layers; layer++){

				s = (sizeof(float) * 1 * dim_qkv) * n_queries;
				model->w.qb[layer] = p;
				p += s;

				s = (sizeof(float) * 1 * dim_qkv) * n_keys;
				model->w.kb[layer] = p;
				p += s;

				s = (sizeof(float) * 1 * dim_qkv) * n_keys;
				model->w.vb[layer] = p;
				p += s;
			}


	
	
			//OUTPUT matrixes (one for each head/query)
			//size is the SAME as QUERY matrixes;    BUT:
			//1) they are (dim_qkv,dim_stream) instead of (dim_stream,dim_qkv)
			//2) their lines are interleaved, because we are going to do a single matmul to get the output projection from the concatenation of all the heads outputs
			s = (sizeof(float) * dim_stream * dim_qkv) * n_queries;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.om[layer] = p;
				p += s;
			}
			//OUTPUT prjection bias
			s = (sizeof(float) * dim_stream * 1) * 1;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.ob[layer] = p;
				p += s;
			}
	
	
	
	
			//POST-NORM: weights
			if( (model->config.norm_cfg[1] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[1] == NORM_RMSNORM)		//RMSNORM
			){
				s = sizeof(float) * dim_stream;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_post_w[layer] = p;
					p += s;
				}
			}
	
	
			//POST-NORM: biases
			if( (model->config.norm_cfg[1] == NORM_LAYERNORM)	//LAYERNORM
			){
				s = sizeof(float) * dim_stream;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_post_b[layer] = p;
					p += s;
				}
			}
	
	
	
			//LINEAR LAYER PRE-FeedForwardNetwork
	
			//weights
			s = sizeof(float) * dim_stream * hidden_dim;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.pre_ffn_w[layer] = p;
				p += s;
			}	
		
			//biases, if required
			if(model->config.bias_cfg[0] == BIAS_ON){
				s = sizeof(float) * hidden_dim;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.pre_ffn_b[layer] = p;
					p += s;
				}
			}
	

	
			//LINEAR LAYER POST-Feed Forward Network
	
			//weights
			s = sizeof(float) * dim_stream * hidden_dim;
			for(int layer = 0; layer < n_layers; layer++){
				model->w.post_ffn_w[layer] = p;
				p += s;
			}
			
			//biases, if required
			if(model->config.bias_cfg[1] == BIAS_ON){
				s = sizeof(float) * dim_stream;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.post_ffn_b[layer] = p;
					p += s;
				}
			}


							
			//ONE STEP BACKWARD: INSIDE the Feed Forward Network itself:
			//no special weights are required here; all the mess is in the two linear layers (pre+post)
			//... UNLESS ...
			//special case: e.g.: LLAMA2 SILU non-linearity, which requires additional multipliers
			if(	(model->config.ffn_nl_type[1] == GATE_ON)
				||
				(model->config.ffn_nl_type[0] == NL_SILU_LLAMA)
			){
				s = sizeof(float) * dim_stream * hidden_dim;
				for(int layer = 0; layer < n_layers; layer++){
					model->w.pre_ffn_w2[layer] = p;
					p += s;
				}
			}
	
	
	
			//FINAL-NORM: weights
			if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[2] == NORM_RMSNORM)		//RMSNORM
			){
				s = sizeof(float) * dim_stream;
				model->w.norm_final_w = p;
				p += s;
			}




			//FINAL-NORM: biases
			if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
			){
				s = sizeof(float) * dim_stream;
				model->w.norm_final_b = p;
				p += s;
			}
	
	
	
			//FINAL LINEAR LAYER (classifier to logits)
	
			//weights
			if(model->config.embeddings_cfg == EMBEDDINGS_SHARED){	//GPT-2 has tied embeddings
				model->w.logits_classifier = model->w.embeddings;
			}
			else
			if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){
				s = sizeof(float) * dim_stream * vocab_size;
				model->w.logits_classifier = p;
				p += s;
			}

		}
		else
		if(checkpoint_type == CP_SAFETENSORS){	//checkpoint file(s) in SafeTensors format

			char * prefix;	//this is the prefix of the part of the model inside the whole safetensors package, 
					//e.g.: "language_model.model", "model", "vision_tower.vision_model.encoder"




			//model-type-specific tensors
			
			if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){

				if(model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL){

					prefix = "vision_tower.vision_model.embeddings.";

					model->w.vision_embeddings_w = safetensors_get_tensor(model, "vision_embeddings_w", -1, prefix);
					model->w.vision_embeddings_b = safetensors_get_tensor(model, "vision_embeddings_b", -1, prefix);
					model->w.learned_pose_w      = safetensors_get_tensor(model, "learned_pose_w",       -1, prefix);

					prefix = "vision_tower.vision_model.";

					//FINAl-NORM: weights
					if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
					    ||
					    (model->config.norm_cfg[2] == NORM_RMSNORM)		//RMSNORM
					){
						model->w.norm_final_w = safetensors_get_tensor(model, "norm_final_w", -1, prefix); 
					}



					//FINAL-NORM: biases
					if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
					){
						model->w.norm_final_b = safetensors_get_tensor(model, "norm_final_b", -1, prefix); 
					}
	
	

					prefix = "multi_modal_projector.";

					model->w.multimodal_projector_w = safetensors_get_tensor(model, "multimodal_projector_w", -1, prefix);
					model->w.multimodal_projector_b = safetensors_get_tensor(model, "multimodal_projector_b", -1, prefix);

				}
				else{

					mylog(LOG_ERROR, "don't know how to handle %s model type with %s architecture when loading tensors. Exiting...",
						modeltype_text[model->config.submodel_type], arch_text[model->config.architectures] 
					);
					exit(1);
				}
			}
			else
			if(model->config.submodel_type == MODELTYPE_DECODER){


				if( (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)
				){

					prefix = "language_model.model.";
				}
				else
				if( (model->config.architectures == ARCH_LLAMA_CAUSAL)
				    ||
				    (model->config.architectures == ARCH_GEMMA_CAUSAL)
				){

					prefix = "model.";
				}
				else{

					mylog(LOG_ERROR, "don't know how to handle %s model type with %s architecture when loading tensors. Exiting...",
						modeltype_text[model->config.submodel_type], arch_text[model->config.architectures] 
					);
					exit(1);
				}



				//FINAl-NORM: weights
				if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
				    ||
				    (model->config.norm_cfg[2] == NORM_RMSNORM)		//RMSNORM
				){
					model->w.norm_final_w = safetensors_get_tensor(model, "norm_final_w", -1, prefix); 
				}

				//FINAL-NORM: biases
				if( (model->config.norm_cfg[2] == NORM_LAYERNORM)	//LAYERNORM
				){
					model->w.norm_final_b = safetensors_get_tensor(model, "norm_final_b", -1, prefix); 
				}


	
				model->w.embeddings = safetensors_get_tensor(model, "embeddings", -1, prefix);

				if(model->config.embeddings_cfg == EMBEDDINGS_SHARED){
					model->w.logits_classifier = model->w.embeddings;
				}
				else
				if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){

					//WARNING: don't use "prefix" here!! The full tensor name is just "lm_head.weight" !!
					model->w.logits_classifier = safetensors_get_tensor(model, "logits_classifier", -1, "");
					//model->w.logits_classifier = safetensors_get_tensor(model, "logits_classifier", -1, prefix);
					
				}

			}



			//tensors for all model types	

			if(model->config.submodel_type == MODELTYPE_VISION_ENCODER){

				if(model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL){

					prefix = "vision_tower.vision_model.encoder.";
				}
			}
			else
			if(model->config.submodel_type == MODELTYPE_DECODER){


				if( (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)
				){

					prefix = "language_model.model.";
				}
				else
				if( (model->config.architectures == ARCH_LLAMA_CAUSAL)
				    ||
				    (model->config.architectures == ARCH_GEMMA_CAUSAL)
				){

					prefix = "model.";
				}

			}
	




			//PRE-NORM: weights
			if( (model->config.norm_cfg[0] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[0] == NORM_RMSNORM)		//RMSNORM
			){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_pre_w[layer] = safetensors_get_tensor(model, "norm_pre_w", layer, prefix);
				}
			}
	
	
			//PRE-NORM: biases
			if( (model->config.norm_cfg[0] == NORM_LAYERNORM)	//LAYERNORM
			){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_pre_b[layer] = safetensors_get_tensor(model, "norm_pre_b", layer, prefix); 
				}
			}
	
	
	
	
	
			//QUERY matrixes
			for(int layer = 0; layer < n_layers; layer++){
				model->w.qm[layer] = safetensors_get_tensor(model, "qm", layer, prefix); 
			}
	
			//KEY matrixes
			for(int layer = 0; layer < n_layers; layer++){
				model->w.km[layer] = safetensors_get_tensor(model, "km", layer, prefix);
			}
			
			//VALUE matrixes
			//size is the SAME as KEY matrixes: 
			for(int layer = 0; layer < n_layers; layer++){
				model->w.vm[layer] = safetensors_get_tensor(model, "vm", layer, prefix);
			}
	
			//OUTPUT matrixes (one for each head/query)
			//size is the SAME as QUERY matrixes;    BUT:
			//1) they are (dim_qkv,dim_stream) instead of (dim_stream,dim_qkv)
			//2) their lines are interleaved, because we are going to do a single matmul to get the output projection from the concatenation of all the heads outputs
			for(int layer = 0; layer < n_layers; layer++){
				model->w.om[layer] = safetensors_get_tensor(model, "om", layer, prefix); 
			}
	

	
			if(model->config.bias_cfg[3] == BIAS_ON){

				//QUERY biases
				for(int layer = 0; layer < n_layers; layer++){
					model->w.qb[layer] = safetensors_get_tensor(model, "qb", layer, prefix); 
				}
	
				//KEY biases
				for(int layer = 0; layer < n_layers; layer++){
					model->w.kb[layer] = safetensors_get_tensor(model, "kb", layer, prefix);
				}
			
				//VALUE biases
				for(int layer = 0; layer < n_layers; layer++){
					model->w.vb[layer] = safetensors_get_tensor(model, "vb", layer, prefix);
				}
	
				//OUTPUT biases
				for(int layer = 0; layer < n_layers; layer++){
					model->w.ob[layer] = safetensors_get_tensor(model, "ob", layer, prefix); 
				}
	

			}
	
	
			//POST-NORM: weights
			if( (model->config.norm_cfg[1] == NORM_LAYERNORM)	//LAYERNORM
			    ||
			    (model->config.norm_cfg[1] == NORM_RMSNORM)		//RMSNORM
			){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_post_w[layer] = safetensors_get_tensor(model, "norm_post_w", layer, prefix); 
				}
			}
	
	
			//POST-NORM: biases
			if( (model->config.norm_cfg[1] == NORM_LAYERNORM)	//LAYERNORM
			){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.norm_post_b[layer] = safetensors_get_tensor(model, "norm_post_b", layer, prefix); 
				}
			}
	
	
	
			//LINEAR LAYER PRE-FeedForwardNetwork
	
			//weights
			for(int layer = 0; layer < n_layers; layer++){
				model->w.pre_ffn_w[layer] = safetensors_get_tensor(model, "pre_ffn_w", layer, prefix); 
			}	
		
			//biases, if required
			if(model->config.bias_cfg[0] == BIAS_ON){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.pre_ffn_b[layer] = safetensors_get_tensor(model, "pre_ffn_b", layer, prefix); 
				}
			}
	

	
			//LINEAR LAYER POST-Feed Forward Network
	
			//weights
			for(int layer = 0; layer < n_layers; layer++){
				model->w.post_ffn_w[layer] = safetensors_get_tensor(model, "post_ffn_w", layer, prefix);
			}
			
			//biases, if required
			if(model->config.bias_cfg[1] == BIAS_ON){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.post_ffn_b[layer] = safetensors_get_tensor(model, "post_ffn_b", layer, prefix);
				}
			}


							
			//ONE STEP BACKWARD: INSIDE the Feed Forward Network itself:
			//no special weights are required here; all the mess is in the two linear layers (pre+post)
			//... UNLESS ...
			//special case: e.g.: LLAMA2 SILU non-linearity, which requires additional multipliers
			if(	(model->config.ffn_nl_type[1] == GATE_ON)
				||
				(model->config.ffn_nl_type[0] == NL_SILU_LLAMA)
			){
				for(int layer = 0; layer < n_layers; layer++){
					model->w.pre_ffn_w2[layer] = safetensors_get_tensor(model, "pre_ffn_w2", layer, prefix); 
				}
			}
	
	
	
			//FINAl-NORM: weights & bases
			
			//see above, in the "model-type-specific tensors" section
	
	
	
			//FINAL LINEAR LAYER (classifier to logits)
	
			//see above, in the "model-type-specific tensors" section

		}

	}
	else
	if(action == ACTION_CREATE){

		//now, let's allocate and initialize all model weights from scratch,
        	//according to the specified configuration.
	        init_weights(model);
		
	}



	//VERY IMPORTANT NOTE:
	//kv_cache must be allocated separately, because it may not be (fully) required (e.g: during training steps)

}


// ============================================================
//  Tokenizer
// ============================================================

int compare_tokens(const void * a, const void * b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void mymemcpy(byte * out, byte * in, size_t len){
	for(size_t i = 0; i < len; i++){
		out[i] = in[i];
	}
}

void build_tokenizer(Model * model, Tokenizer * t, char * tokenizer_path, int vocab_size){

	t->byte_fallback_offset = -1;	//in the vocabulary, this is the id of the token containing byte 0x00, being the first single-byte token

	t->singlebytes_space_firstid = -1;
	t->singlebytes_space_lastid = -1;

	t->addedtokens_space_firstid = -1;
	t->addedtokens_space_lastid = -1;


	
	if(	(tokenizer_format == TOKFORMAT_TRIP)
		||
		(tokenizer_format == TOKFORMAT_LLAMA2_AK)
	){


		t->singlebytes_space_firstid = 3;
		t->singlebytes_space_lastid  = 3 + 255;	//+256 would be the first token after the last one in the bytespace

		t->addedtokens_space_firstid = 0;
		t->addedtokens_space_lastid  = 2;	//id of last token in the added_tokens space




	    FILE * f;
	
	    if(tokenizer_path != NULL){
	
	
	 	//read in the file
		f = fopen(tokenizer_path, "rb");
	
	    	if(f==NULL){
		    mylog(LOG_ERROR, "couldn't load %s", tokenizer_path); 
		    exit(EXIT_FAILURE);
	    	}
		
	
		int temp_vocab_size = vocab_size;
	
		if(tokenizer_format == TOKFORMAT_TRIP){	//TRIP tokenizer file type has an additional entry at the very top, containing the size of the vocabulary
	    		if(fread(&temp_vocab_size, sizeof(int), 1, f) != 1){
				mylog(LOG_ERROR, "failed read (build_tokenizer: temp_vocab_size"); 
				exit(EXIT_FAILURE); 
	    		}
	
			if(temp_vocab_size > MAX_VOCAB_ENTRIES){
	
				mylog(LOG_ERROR,"vocab_size specified in tokenizer file (%d) is larger than TRIP hard-coded maximum vocab_size (%d). Exiting",temp_vocab_size,MAX_VOCAB_ENTRIES); 
				exit(EXIT_FAILURE); 
	    	
			}
			
			
			vocab_size = temp_vocab_size;
	    		mylog(LOG_INFO,"vocab_size (tokenizer) = %d",vocab_size);
		}
	
	
	    }
	
	    t->vocab_size = vocab_size;

	    t->pad_id = 0;
	    t->bos_id = 1;
	    t->eos_id = 2;
	    t->byte_fallback_offset = 3;	//that's 3 for LLAMA
	
	    // calloc space to hold the scores and the strings
	    t->vocab = (char**)calloc(vocab_size, sizeof(char*));
	    t->vocab_scores = (float*)calloc(vocab_size, sizeof(float));
	    t->sorted_vocab = NULL; //we will initialize this later, when we need to build the sorted vocabulary
	    //for (int i = 0; i < 256; i++) {
	    //    t->byte_pieces[i * 2] = (unsigned char)i;
	    //    t->byte_pieces[i * 2 + 1] = '\0';
	    //}
	    
	    //if we are not passing a file, just stop here initialization; we will build the vocabulary afterwards
	    if(tokenizer_path==NULL)	return;
	    
	    //read in the file
	    if(fread(&t->max_token_length, sizeof(int), 1, f) != 1){
		    mylog(LOG_ERROR,"failed read (build_tokenizer: max_token_length)"); 
		    exit(EXIT_FAILURE); 
	    }
	
	
	    for(int i = 0; i < vocab_size; i++){
	
	    	int len;
	
	        if(fread(t->vocab_scores + i, sizeof(float), 1, f) != 1){
			mylog(LOG_ERROR, "failed read (build_tokenizer: vocab_scores %d)",i); 
			exit(EXIT_FAILURE);
		}
	        if(fread(&len, sizeof(int), 1, f) != 1){
			mylog(LOG_ERROR, "failed read (build_tokenizer: len %d)",i); 
			exit(EXIT_FAILURE); 
		}
	
	        t->vocab[i] = (char *)calloc((len + 1), sizeof(char));
	
	        if(fread(t->vocab[i], sizeof(char), len, f) != len){
			mylog(LOG_ERROR, "failed read (build_tokenizer: vocab %d)",i);
			exit(EXIT_FAILURE); 
		}
	
	        t->vocab[i][len] = '\0'; // add the string terminating token
	    }
	
	    fclose(f);
	
	}
	else
	if(tokenizer_format == TOKFORMAT_GPT2_AK)
	{
	
	    FILE * f;
	
	    if(tokenizer_path != NULL){
	
	
	 	//read in the file
		f = fopen(tokenizer_path, "rb");
	
	    	if(f==NULL){
		    mylog(LOG_ERROR, "couldn't load %s", tokenizer_path); 
		    exit(EXIT_FAILURE);
	    	}
		
	
		int temp_vocab_size = vocab_size;
	    }
    
	    //if we are not passing a file, just stop here initialization; we will build the vocabulary afterwards
	    if(tokenizer_path==NULL)	return;


	    uint32_t header[256];
	    if(fread(header, sizeof(uint32_t), 256, f) != 256){
		mylog(LOG_ERROR,"cannot read header from tokenizer file \"%s\". Exiting...",tokenizer_path);	exit(-1);
	    }
	    if(header[0] != 20240328){
		mylog(LOG_VERBOSE_DEBUG,"invalid header in tokenizer file \"%s\". Exiting...",tokenizer_path);	exit(-1);
	    }
	
	    int version = header[1];
	    t->vocab_size = header[2];	vocab_size = t->vocab_size;
	
	    if(version == 1){
		t->eos_id = 50256;
		if(t->vocab_size != 50257){
			mylog(LOG_VERBOSE_DEBUG,"invalid vocabulary size %d in tokenizer file \"%s\". Exiting...",header[2],tokenizer_path);  exit(-1);
		}
	    }
	    else
	    if(version == 2){
		t->eos_id = header[3];
	    }
	    else{
		mylog(LOG_VERBOSE_DEBUG,"invalid header version %d in tokenizer file \"%s\". Exiting...",header[1],tokenizer_path);  exit(-1);
	    }
	
	    //t->pad_id = 0;
	    //t->bos_id = 1;
	    //t->eos_id = 2;
	    //t->byte_fallback_offset = 3;	//that's 3 for LLAMA
	

	    // calloc space to hold the scores and the strings
	    t->vocab = (char**)calloc(vocab_size, sizeof(char*));
	    t->vocab_scores = (float*)calloc(vocab_size, sizeof(float));
	    t->sorted_vocab = NULL; //we will initialize this later, when we need to build the sorted vocabulary
	    //for (int i = 0; i < 256; i++) {
	    //    t->byte_pieces[i * 2] = (unsigned char)i;
	    //    t->byte_pieces[i * 2 + 1] = '\0';
	    //}


	/*	    
	    //read in the file
	    if(fread(&t->max_token_length, sizeof(int), 1, f) != 1){
		    mylog(LOG_ERROR,"failed read (build_tokenizer: max_token_length)"); 
		    exit(EXIT_FAILURE); 
	    }
	*/
	
	    for(int i = 0; i < vocab_size; i++){
	
	    	int len = 0x00000000;

	/*	
	        if(fread(t->vocab_scores + i, sizeof(float), 1, f) != 1){
			mylog(LOG_ERROR, "failed read (build_tokenizer: vocab_scores %d)",i); 
			exit(EXIT_FAILURE);
		}
	*/
	        if(fread(&len, sizeof(unsigned char), 1, f) != 1){
			mylog(LOG_ERROR, "failed read (build_tokenizer: len @ #%d/%d)",i,vocab_size); 
			exit(EXIT_FAILURE); 
		}
	
	        t->vocab[i] = (char *)calloc((len + 1), sizeof(char));
	
	        if(fread(t->vocab[i], sizeof(char), len, f) != len){
			mylog(LOG_ERROR, "failed read (build_tokenizer: vocab %d - couldn't read len %d)",i,len);
			exit(EXIT_FAILURE); 
		}
	
	        t->vocab[i][len] = '\0'; // add the string terminating token
	    }
	
	    fclose(f);
	
	}
	else
        if(tokenizer_format == TOKFORMAT_JSON_HUGGINGFACE){

	    int temp_vocab_size = vocab_size;

	    t->vocab_size = vocab_size;


	    // calloc space to hold the scores and the strings
	    t->vocab = (char**)calloc(vocab_size, sizeof(char*));
	    t->vocab_scores = (float*)calloc(vocab_size, sizeof(float));
	    t->sorted_vocab = NULL; //we will initialize this later, when we need to build the sorted vocabulary
	    //for (int i = 0; i < 256; i++) {
	       //	t->byte_pieces[i * 2] = (unsigned char)i;
	       //	t->byte_pieces[i * 2 + 1] = '\0';
	    //}
	    
	    //if we are not passing a file, just stop here initialization; we will build the vocabulary afterwards
	    if(tokenizer_path==NULL){
		return;
	    } 


	    FILE * f;
	
	    if(tokenizer_path != NULL){
	
	
	 	//read in the file
		f = fopen(tokenizer_path, "rb");
	
	    	if(f==NULL){
		    mylog(LOG_ERROR, "couldn't load %s", tokenizer_path); 
		    exit(EXIT_FAILURE);
	    	}
		
	
		ssize_t size;
	
		fseek(f,0,SEEK_END);
		size = ftell(f);
		fclose(f);

		if(size<0){
			mylog(LOG_ERROR,"Tokenizer file \"%s\" size is invalid (<0). Exiting...",tokenizer_path); exit(-1); 
		}


		unsigned char * json;

		int fd = open(tokenizer_path, O_RDONLY);

		if(fd < 0){
			mylog(LOG_ERROR,"Cannot open tokenizer file \"%s\". Exiting...",tokenizer_path); exit(-1);
		}
	

		json = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

		if(json == MAP_FAILED){
			mylog(LOG_ERROR,"Cannot mmap the tokenizer file \"%s\". Exiting...",tokenizer_path); exit(-1);
		}

		
		JsonNode * tokenizer_root = NULL;
		char * p = json;	//we must preserve "json" pointer value
		tokenizer_root = parseJsonValue(&p);

		
		if(log_cfg >= LOG_DEBUG){
			//printJsonTree(tokenizer_root, 0);
		}

	
		JsonNode * tokenizer_model = NULL;
		tokenizer_model = findJsonNodeByKey(tokenizer_root, "model");

		if(tokenizer_model==NULL){
			mylog(LOG_ERROR, "Invalid tokenizer file \"%s\" for specified tokenizer format \"%s\": key \"model\" not found. Exiting...", tokenizer_path, tokformat_text[tokenizer_format]);
			exit(-1);
		}

		JsonNode * tokenizer_model_type = NULL;
		tokenizer_model_type = findJsonNodeByKey(tokenizer_model, "type");

		if(tokenizer_model_type==NULL){
			mylog(LOG_ERROR, "Invalid tokenizer file \"%s\" for specified tokenizer format \"%s\": key \"type\" not found. Exiting...", tokenizer_path, tokformat_text[tokenizer_format]);
			exit(-1);
		}

		char * type = tokenizer_model_type->value.stringValue; 
		if(type == NULL){
			mylog(LOG_ERROR, "Invalid tokenizer file \"%s\" for specified tokenizer format \"%s\": key \"type\" is not a string. Exiting...", tokenizer_path, tokformat_text[tokenizer_format]);
			exit(-1);
	
		}

		//let's check that the tokenizer type is supported
		if((strcmp(type,"BPE")==0) || (strcmp(type,"bpe")==0)){
			//do nothing
		}
		else{
			mylog(LOG_ERROR, "Tokenizer file \"%s\" specifies tokenizer type \"%s\" which is not supported. Exiting...", tokenizer_path, type);
			exit(-1);
		}



		JsonNode * tokenizer_model_vocab = NULL;
		tokenizer_model_vocab = findJsonNodeByKey(tokenizer_model, "vocab");

		if(tokenizer_model==NULL){
			mylog(LOG_ERROR, "Invalid tokenizer file \"%s\" for specified tokenizer format \"%s\": key \"vocab\" not found. Exiting...", tokenizer_path, tokformat_text[tokenizer_format]);
			exit(-1);
		}


		//let's initialize max_token_length
		t->max_token_length = 1;

		//here we load the vocabulary
		JsonNode * curr_tok = tokenizer_model_vocab->value.children;
		while(curr_tok != NULL){

			//let's load this token

			//here the format is: 
			// - the text of the token is the key/label
			// - the value related to the key/label is an integer which is the token id

		    	int len;
			int i,j;
			char * tok_text = curr_tok->key;
			unsigned char bytebuf[2];
			unsigned char tokbuf[4096];

			i = atoi(curr_tok->value.stringValue);
			len = strlen(tok_text);

			strcpy(tokbuf,curr_tok->key);

			//WARNING: translate special space sequence U+2581 (▁) to simple space
			//THIS MAY HAVE TO BE CHANGED IN THE FUTURE, and HANDLED DIFFERENTLY!!!
			j = 0;
			while(j<len){
				if(memcmp(&tokbuf[j],"\xE2\x96\x81",3)==0){
					mymemcpy(&tokbuf[j],&tokbuf[j+2],(len+1)-(j+2));
					tokbuf[j] = ' ';
					len -= 2;
				}
				j++;
			}			
			
			tok_text = tokbuf;			

			int hex = -1;
			if(sscanf(tok_text, "<0x%02X>", &hex) == 1){	//if the token represents a single byte...
				len = 1;
				tok_text = &bytebuf[0];
				bytebuf[0] = (unsigned char)hex;
				bytebuf[1] = '\0';

				if(hex == 0x00)	t->byte_fallback_offset = i;

				if(t->singlebytes_space_firstid == -1)	t->singlebytes_space_firstid = i;

				t->singlebytes_space_lastid = i;

			}
			else{
				if(model->config.architectures == ARCH_LLAMA_CAUSAL){
					if(strcmp(tok_text, "<unk>")==0)	t->pad_id = i;
					else
					if(strcmp(tok_text, "<s>")==0)		t->bos_id = i;
					else
					if(strcmp(tok_text, "</s>")==0)		t->eos_id = i;
				}
				else{
					if(strcmp(tok_text, "<pad>")==0)	t->pad_id = i;
					else
					if(strcmp(tok_text, "<bos>")==0)	t->bos_id = i;
					else
					if(strcmp(tok_text, "<eos>")==0)	t->eos_id = i;
				}
				
			}


			if(len > t->max_token_length)	t->max_token_length = len;
	
		        t->vocab_scores[i] = 0.0;
		        t->vocab[i] = (char *)calloc((len + 1), sizeof(char));
			strcpy(t->vocab[i], tok_text);	

	
//printf("%d: %s\n",i,t->vocab[i]);


			//go to next entry
			curr_tok = curr_tok->next;
		}



		
		JsonNode * tokenizer_added_tokens = NULL;
		tokenizer_added_tokens = findJsonNodeByKey(tokenizer_root, "added_tokens");

		//inside here, we load the extra tokens added to the vocabulary
		if(tokenizer_added_tokens  !=  NULL){

			//it's an array of objects: let's step into the array, first element
			JsonNode * curr_tok = tokenizer_added_tokens->value.children;

			while(curr_tok != NULL){

				//here the format is DIFFERENT:
				// - there's a key "id" whose value is an integer which is the token id
				// - there's a key "content" whose value is a string with the text of the token

			    	int len;
				int i;

				JsonNode * id = NULL;
				JsonNode * content = NULL;

				id = findJsonNodeByKey(curr_tok, "id");
				content = findJsonNodeByKey(curr_tok, "content");

				if((id!=NULL) && (content!=NULL)){

					i = atoi(id->value.stringValue);
					len = strlen(content->value.stringValue);
	
					if(len > t->max_token_length)	t->max_token_length = len;

				        t->vocab_scores[i] = 0.0;
			        	t->vocab[i] = (char *)calloc((len + 1), sizeof(char));
					strcpy(t->vocab[i], content->value.stringValue);	

					if(t->addedtokens_space_firstid == -1)	t->addedtokens_space_firstid = i;
					t->addedtokens_space_lastid = i;


/*	
printf("%d: %s\n",i,t->vocab[i]);
fflush(stdout);
*/

				}


				//go to next entry
				curr_tok = curr_tok->next;
			}
	
		}


		freeJsonTree(tokenizer_root);

		//OK, let's close the tokenizer file: we've had enough!
		munmap(json, size);
		close(fd);

	   }	


	   //NOW, we perform a final check, because some tokenizer.json files omit some tokens (!!!), thus some tokens may be uninitialized

	   int unused_idx = 1;

	   for(int i=0; i < vocab_size ; i++){

		if( t->vocab[i] == NULL ){
			char unused_token[32];
			sprintf(unused_token,"<unused_TRiP_%06d>",unused_idx);

		        t->vocab_scores[i] = 0.0;
	        	t->vocab[i] = (char *)calloc((strlen(unused_token) + 1), sizeof(char));
			strcpy(t->vocab[i], unused_token);	

			unused_idx++;
		}
	   }

	}



/*
	if(model->config.architectures == ARCH_LLAMA_CAUSAL){

		t->singlebytes_space_firstid = 3;
		t->singlebytes_space_lastid  = 3 + 255;	//+256 would be the first token after the last one in the bytespace

		t->addedtokens_space_firstid = 0;
		t->addedtokens_space_lastid  = 2;	//id of last token in the added_tokens space
	}
	else
	if( (model->config.architectures == ARCH_GEMMA_CAUSAL)
	    ||
	    (model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)
	){
		t->singlebytes_space_firstid = 217;
		t->singlebytes_space_lastid  = 217 + 255;	//+256 would be the first token after the last one in the bytespace

		t->addedtokens_space_firstid = 0;
		t->addedtokens_space_lastid  = 216;	//id of last token in the added_tokens space
	}
	else{
		t->singlebytes_space_firstid = -1;
		t->singlebytes_space_lastid  = -1;

		t->addedtokens_space_firstid = -1;
		t->addedtokens_space_lastid  = -1;	//id of last token in the added_tokens space
	}
*/


        //mapping of added_tokens is sparse in this case; let's explicitly provide the correct ranges
	if(model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL)        {
                t->singlebytes_space_firstid = 217;
                t->singlebytes_space_lastid  = 217 + 255;       //+256 would be the first token after the last one in the bytespace

                t->addedtokens_space_firstid = 0;
                t->addedtokens_space_lastid  = 216;     //id of last token in the added_tokens space
        }



/*
	if(log_cfg==LOG_VERBOSE_DEBUG){
		dump_vocab(t);
	}
*/

}



//helper function to escape special characters for JSON strings
void escape_json_string(char * input, char * output){
    int j = 0;
    //allocate a buffer that is definitely large enough (worst case: every char is escaped)
    char * temp_out = output;
    while(*input){
        switch(*input){
            case '\"': *temp_out++ = '\\'; *temp_out++ = '\"'; break;
            case '\\': *temp_out++ = '\\'; *temp_out++ = '\\'; break;
            case '\b': *temp_out++ = '\\'; *temp_out++ = 'b'; break;
            case '\f': *temp_out++ = '\\'; *temp_out++ = 'f'; break;
            case '\n': *temp_out++ = '\\'; *temp_out++ = 'n'; break;
            case '\r': *temp_out++ = '\\'; *temp_out++ = 'r'; break;
            case '\t': *temp_out++ = '\\'; *temp_out++ = 't'; break;
            default:
                //one should make sure to only copy printable ASCII or valid UTF-8 multi-byte sequences;
                //this simple version will suffice if vocab is clean.
                *temp_out++ = *input;
                break;
        }
        input++;
    }
    *temp_out = '\0';
}



//function to save tokenizer in HuggingFace JSON format
void save_tokenizer_json(Tokenizer * t, char * filepath, int vocab_size){

    FILE * f = fopen(filepath, "w");
    if(f == NULL){
        mylog(LOG_ERROR, "couldn't open %s for write. Exiting...", filepath);
        exit(-1);
    }

    mylog(LOG_INFO, "Saving tokenizer to %s (HuggingFace JSON format)...", filepath);

    //buffer for escaping token strings
    char * escaped_token = (char *)calloc(((t->max_token_length * 2) + 1), sizeof(char));

    fprintf(f, "{\n");
    fprintf(f, "  \"version\": \"1.0\",\n");
    fprintf(f, "  \"truncation\": null,\n");
    fprintf(f, "  \"padding\": null,\n");

    // added_tokens section (for special tokens)
    fprintf(f, "  \"added_tokens\": [\n");
    if((t->pad_id < vocab_size)  &&  (t->vocab[t->pad_id]!=NULL)){
        escape_json_string(t->vocab[t->pad_id], escaped_token);
        fprintf(f, "    {\"id\": %u, \"content\": \"%s\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}", t->pad_id, escaped_token);
    }
    if((t->bos_id < vocab_size)  &&  (t->vocab[t->bos_id]!=NULL)){
        escape_json_string(t->vocab[t->bos_id], escaped_token);
        fprintf(f, ",\n    {\"id\": %u, \"content\": \"%s\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}", t->bos_id, escaped_token);
    }
    if((t->eos_id < vocab_size)  &&  (t->vocab[t->eos_id]!=NULL)){
        escape_json_string(t->vocab[t->eos_id], escaped_token);
        fprintf(f, ",\n    {\"id\": %u, \"content\": \"%s\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}", t->eos_id, escaped_token);
    }
    fprintf(f, "\n  ],\n");


    //main vocabulary
    fprintf(f, "  \"model\": {\n");
    fprintf(f, "    \"type\": \"BPE\",\n");
    fprintf(f, "    \"vocab\": {\n");

    for(int i = 0; i < vocab_size; i++){
        escape_json_string(t->vocab[i], escaped_token);
        fprintf(f, "      \"%s\": %d", escaped_token, i);
        if(i < vocab_size - 1){
            fprintf(f, ",\n");
        } else {
            fprintf(f, "\n");
        }
    }

    fprintf(f, "    }\n");
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    free(escaped_token);
    mylog(LOG_INFO, "Tokenizer successfully saved.");
}



void save_tokenizer(Tokenizer * t, char * tokenizer_path, int vocab_size){

    if(tokenizer_format == TOKFORMAT_JSON_HUGGINGFACE){
        save_tokenizer_json(t, tokenizer_path, vocab_size);
        return;
    }



    //read in the file
    FILE * f = fopen(tokenizer_path, "wb");
    if(f==NULL){
	    mylog(LOG_ERROR,"couldn't open %s for write", tokenizer_path); 
	    exit(EXIT_FAILURE);
    }

    if(tokenizer_format == TOKFORMAT_TRIP){	//TRIP tokenizer file type has an additional entry at the very top, containing the size of the vocabulary
    	if(fwrite(&vocab_size, sizeof(int), 1, f) != 1){
	    mylog(LOG_ERROR, "failed write"); 
	    exit(EXIT_FAILURE); 
    	}
    }

    if(fwrite(&t->max_token_length, sizeof(int), 1, f) != 1){
	    mylog(LOG_ERROR, "failed write"); 
	    exit(EXIT_FAILURE); 
    }


    for(int i = 0; i < vocab_size; i++){

    	int len;
	len = strlen(t->vocab[i]);

        if(fwrite(t->vocab_scores + i, sizeof(float), 1, f) != 1){
		mylog(LOG_ERROR, "failed write: vocab_scores @ entry %d",i); 
		exit(EXIT_FAILURE);
	}
        if(fwrite(&len, sizeof(int), 1, f) != 1){
		mylog(LOG_ERROR, "failed write: len @ entry %d",i); 
		exit(EXIT_FAILURE); 
	}

	if(fwrite(t->vocab[i], sizeof(char), len, f) != len){
		mylog(LOG_ERROR, "failed write: vocab string @ entry %d",i);
		exit(EXIT_FAILURE); 
	}

    }

    fclose(f);
}


//does the text starting from the str pointer match (at any length) with one of the "added tokens" (which is a set of manually defined special tokens) ?
//this is the role if the pre-tokenizer
int pretokenizer(Model * model, char * str, Tokenizer * t){

	int pre_id = -1;

	int first = t->addedtokens_space_firstid;
	int last  = t->addedtokens_space_lastid;

	if(first != -1){

		int i;
		int str_len = strlen(str);

		for(i = first; i <= last; i++){

			int tok_len = strlen(t->vocab[i]);

			//this one is too long
			if(tok_len > str_len) continue;

			if(memcmp(t->vocab[i], str, tok_len) == 0){
				//we found a special token in the added_tokens list!
				pre_id = i;
				break;
			}
		}
	}

	return pre_id;
}


int str_lookup(Model * model, char * str, Tokenizer * t){

	int vocab_size             =  t->vocab_size;
	TokenIndex * sorted_vocab  =  t->sorted_vocab;

	//efficiently find the perfect match for str in vocab, return its index or -1 if not found
	TokenIndex tok = { .str = str }; // acts as the key to search for
	TokenIndex * res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);



	
	if((res!=NULL) && (t->singlebytes_space_firstid != -1)){

		//if the match found is within the vocabulary space for raw single bytes, let's see if there's a non-raw-byte token
		if((res->id >= t->singlebytes_space_firstid)  &&  (res->id <= t->singlebytes_space_lastid)){

			int res_i = (res - &(sorted_vocab[0]));	//this is the position of the found token in the sorted array

			//printf("\nSearching better token for %05d:ID%05d:\"%s\"...",res_i,res->id,res->str);

			if((res_i + 1) < vocab_size){	//if we can look at the next token in the sorted vocabulary (without going outside the vocab space)

				//printf("\nEvaluating %05d:ID%05d:\"%s\"...",(res_i + 1),sorted_vocab[res_i + 1].id,sorted_vocab[res_i + 1].str);

				if(strcmp(sorted_vocab[res_i].str, sorted_vocab[res_i + 1].str) == 0){	//and if the next token is exactly like the raw byte
					res = &sorted_vocab[res_i + 1];	//we choose the next token!
					//printf("\n   NEW TOKEN FOUND: %05d:ID%05d:\"%s\"...",res_i+1,res->id,res->str);
				}
			}
		}
	}

	return res != NULL ? res->id : -1;
}

void text2tokens(Model * model, Tokenizer * t, char * input, unsigned int mode, int * tokens, int * n_tokens){

    	// encode the string text (input) into an upper-bound preallocated tokens[] array
    
	//mode:	0x1:		prepend BOS / <s>  token
	//	0x2:		append  EOS / </s> token
	//	0x4:		prepend single space at the very beginning of the text sequence
	//	0x8:		append "\n" at the end of the text, if not present
	//	0x10-0x40000000	FREE
	//	0x80000000	encode from file (char * input is the path of the file); otherwise, char * input is directly the input stream to be encoded
	//


    if(input == NULL) { mylog(LOG_ERROR, "cannot encode from NULL path"); exit(EXIT_FAILURE); }


    if((model != NULL) && (model->config.architectures == ARCH_LLAMA_CAUSAL))	mode |= 0x4;



    //if we need to alloc and sort the vocabulary
    if (t->sorted_vocab == NULL) {

        t->sorted_vocab = calloc(t->vocab_size, sizeof(TokenIndex));

        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }


        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);


/*
printf("\n\nSORTED VOCAB:\n\n");
for (int i = 0; i < t->vocab_size; i++) {
           printf("i=%05d: id=%05d: %s\n", i, t->sorted_vocab[i].id, t->sorted_vocab[i].str);
}
exit(1);
*/

    }


    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char * str_buffer = calloc((t->max_token_length*2 +1 +2), sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
//    *n_tokens = 0;		//no! If we are chatting, we may have already tokenized (and decoded) the first part of the sequence
    int start_token_idx = *n_tokens;


    // add optional BOS (=1 in LLAMA) token, if desired
    if(mode&0x1) tokens[(*n_tokens)++] = t->bos_id;


    //from:A.Karpathy (llama2.c)
    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing

    if((input[0] != '\0') && (start_token_idx==0) && (mode&0x4)){
        int dummy_prefix = str_lookup(model, " ", t);
        tokens[(*n_tokens)++] = dummy_prefix;
    }


    // Code point ↔ UTF-8 conversion
    // First code point     Last code point	Byte 1	  Byte 2	Byte 3	Byte 4
    // U+0000	            U+007F	        0xxxxxxx
    // U+0080	            U+07FF	        110xxxxx	10xxxxxx
    // U+0800	            U+FFFF	        1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	            U+10FFFF            11110xxx	10xxxxxx	10xxxxxx	10xxxxxx




if(mode&0x80000000){	//encode from input file

  FILE * f;

  f = fopen(input,"r");

  if(f==NULL){
    mylog(LOG_ERROR,"Invalid path \"%s\" for input text for encoding",input);
    exit(EXIT_FAILURE);
  }

  char * p;
  char text[4096];

  while((p=fgets(text,(4096-1),f))!=NULL){

    text[4095]='\0';

    // process the raw (UTF-8) byte sequence of the input string
    for(char * c = text; *c != '\0'; c++){


	//PRE-TOKENIZER
	int pre_id = pretokenizer(model, c, t);

	if(pre_id == -1){

        	// reset buffer if the current byte is ASCII or a leading byte
	        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        	// 0x80 is 10000000
	        // in UTF-8, all continuation bytes start with "10" in first two bits

        	//if this byte is not a continuation byte
	        if ((*c & 0xC0) != 0x80) {
        	    // this byte must be either a leading byte (11...) or an ASCII char (0x...)
	            // => reset our location, as we're starting a new UTF-8 codepoint
        	    str_len = 0;

	        }

        	// append the current byte to the buffer
	        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        	str_buffer[str_len] = '\0';



	        // while the next character is a continuation byte, continue appending
        	// but if there are too many of them, just stop to avoid overruning str_buffer size.
	        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
        	    continue;
	        }
	}


	if(str_len > 0){
	
	        //OK: c+1 is not a continuation byte, so we've read in a full codepoint
        	int id = str_lookup(model, str_buffer, t);

	        if(id != -1){
        	    // we found this codepoint in vocab, add it as a token
	            tokens[(*n_tokens)++] = id;
        	}
		else{
        	    // byte_fallback encoding: just encode each byte as a token
	            for (int i=0; i < str_len; i++) {
        	        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + t->byte_fallback_offset;
	            }
        	}
	        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
	}
 

	//if we previously found that at current position there is one of the "added tokens", we need it not to be split by BPE later, but atomically tokenized as one
	if(pre_id != -1){

		tokens[(*n_tokens)++] = pre_id;
		int len = strlen(t->vocab[pre_id]) - 1;	//it's -1 because there will be c++ again because of the for loop
		while((*c!='\0') && (len>0)){
			c++;
			len--;
		}
	}

   }

  }

  fclose(f);

}
else{	//encode input text from command line

    // process the raw (UTF-8) byte sequence of the input string
    for(char * c = input; *c != '\0'; c++){


	//PRE-TOKENIZER
	int pre_id = pretokenizer(model, c, t);

	if(pre_id == -1){

        	// reset buffer if the current byte is ASCII or a leading byte
	        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        	// 0x80 is 10000000
	        // in UTF-8, all continuation bytes start with "10" in first two bits

        	//if this byte is not a continuation byte
	        if ((*c & 0xC0) != 0x80) {
        	    // this byte must be either a leading byte (11...) or an ASCII char (0x...)
	            // => reset our location, as we're starting a new UTF-8 codepoint
        	    str_len = 0;

	        }

        	// append the current byte to the buffer
	        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        	str_buffer[str_len] = '\0';



	        // while the next character is a continuation byte, continue appending
        	// but if there are too many of them, just stop to avoid overruning str_buffer size.
	        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
        	    continue;
	        }
	}


	if(str_len >0){

	        //OK: c+1 is not a continuation byte, so we've read in a full codepoint
        	int id = str_lookup(model, str_buffer, t);

	        if(id != -1){
        	    // we found this codepoint in vocab, add it as a token
	            tokens[(*n_tokens)++] = id;
        	}
		else{
        	    // byte_fallback encoding: just encode each byte as a token
	            for (int i=0; i < str_len; i++) {
        	        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + t->byte_fallback_offset;
	            }
        	}
	        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
	}
 

	//if we previously found that at current position there is one of the "added tokens", we need it not to be split by BPE later, but atomically tokenized as one
	if(pre_id != -1){

		tokens[(*n_tokens)++] = pre_id;
		int len = strlen(t->vocab[pre_id]) - 1;	//it's -1 because there will be c++ again because of the for loop
		while((*c!='\0') && (len>0)){
			c++;
			len--;
		}
	}

   }



}



//MERGE

int this_max_tokens = *n_tokens;

if(tokenizer_type == TOKTYPE_TRIP){
    // merge the best consecutive pair each iteration, according the scores in vocab_scores

    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    while (1) {

	int goon = 0;

        for (int i=start_token_idx; i < (*n_tokens-1); i++) {

	    if(is_separator(t->vocab[tokens[i  ]]) > 0){
		    best_score = -1e10;
		    best_id = -1;
		    best_idx = -1;
		    continue;
	    }
	    if(is_separator(t->vocab[tokens[i+1]]) > 0){
		    if(best_idx == -1){
			//do nothing: will reset on the next cycle, which will have i on the separator, see if above
			//the only thing we do is mark this word as fully consumed, and start from the next separator on the next iteration
		    	if(goon==0){	//but only if there is no potential work still to do before this word!!
				start_token_idx = i+1;
			}
		    }
		    else{
		        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
		        tokens[best_idx] = best_id;
		        // delete token at position best_idx+1, shift the entire sequence back 1
		        for (int z = best_idx+1; z < (*n_tokens-1); z++) {
		            tokens[z] = tokens[z+1];
		        }
		        (*n_tokens)--; // token length decreased
			mylog(LOG_VERBOSE_DEBUG,"total tokens = %06d",*n_tokens);

			goon = 1;
			i--;
		    }
		    continue;
	    }

            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);


            int id = str_lookup(model, str_buffer, t);
            if((id != -1) && (t->vocab_scores[id] > best_score)) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if(goon == 0) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

    }

    // add optional EOS (=2 in LLAMA) token, if desired
    if(mode&0x2) tokens[(*n_tokens)++] = t->eos_id;

    free(str_buffer);
}
else
if(tokenizer_type == TOKTYPE_SENTENCEPIECE){


    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=start_token_idx; i < (*n_tokens-1); i++) {

	    if(	(strlen(t->vocab[tokens[i]])==0)
		||
		(strlen(t->vocab[tokens[i+1]])==0)
	    ){continue;}	//let's not merge with an empty token!

            //let's check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(model, str_buffer, t);
            if((id != -1) && (t->vocab_scores[id] > best_score)){
                //this merge pair exists in vocab! let's record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; //we couldn't find any more pairs to merge, so we're done
        }


mylog(LOG_VERBOSE_DEBUG,"Merge found! id%d + id%d = id%d (\"%s\" + \"%s\" = \"%s\")",tokens[best_idx],tokens[best_idx+1],best_id, t->vocab[tokens[best_idx]],t->vocab[tokens[best_idx+1]],t->vocab[best_id]); 


        //merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
	mylog(LOG_VERBOSE_DEBUG,"total tokens = %06d",*n_tokens);
    }


    // append optional "\n" token when absent, if desired
    if(mode&0x8){
	int id = str_lookup(model, "\\n", t);
        if(id != -1){
        	if(tokens[(*n_tokens)-1] != id){
			tokens[(*n_tokens)++] = id;
		}
	}
    }

    // add optional EOS (=2) token, if desired
    if(mode&0x2) tokens[(*n_tokens)++] = t->eos_id;


    free(str_buffer);

}


    mylog(LOG_DEBUG,"\n");
    mylog(LOG_DEBUG,"Final input tokenization:");
    for(int i = 0; i<*n_tokens; i++){
	mylog(LOG_DEBUG,"%05d:  %05d  %s",i,tokens[i],toki.vocab[tokens[i]]);
    }
    mylog(LOG_DEBUG,"\n");



    //clean-up of previously used token slots
    for(int i = *n_tokens ; i < this_max_tokens; i++){        tokens[i] = toki.pad_id;
    }

}


int add_vocab_entry(Tokenizer * t, char * text, int * size){

	int insize = *size;

	if(*size >= t->vocab_size){
		mylog(LOG_DEBUG,"no-add");
		return -1;	
	}

	t->vocab[*size] = (char *)calloc((strlen(text)+1),sizeof(char));
	t->vocab_scores[*size] = (float)(strlen(text));
	strcpy(t->vocab[*size],text);
	(*size)++;

	mylog(LOG_DEBUG,"%05d: %s",(insize),t->vocab[insize]);

	return insize;
}

/*
int is_separator(char * text){
	int val = -1;

	const char * separators = " ,.-;:_?!\"|+*<>/";
	int seplen = strlen(separators);
		
	for(int i=0; i<seplen; i++){
		if(text[0]==separators[i]){
			val = 1;
			break;
		}
	}

	return val;
}
*/

int is_separator(char * text){
	int val = -1;
	char c = text[0];

	if( 	(!((c>='A')&&(c<='Z'))) &&
		(!((c>='a')&&(c<='z'))) &&
		(!((c>='0')&&(c<='9')))
	){
			val = 1;
	}

	return val;
}

void dump_vocab(Tokenizer * t){

	mylog(LOG_INFO,"dump_vocab:\n\nvocab_size=%d\n",t->vocab_size);

	for(int i=0;i<t->vocab_size;i++){
		mylog(LOG_INFO,"%d: \t%.1f \t%s",i,t->vocab_scores[i],t->vocab[i]);
	}


}

void create_vocab(Tokenizer * t, char * input_path){

	int x;

	// create a brand new vocaulary from the input text
	if(input_path == NULL) { fprintf(stderr, "NULL path as input text... exiting\n"); exit(EXIT_FAILURE); }

	// create a temporary buffer that will store merge candidates of always two consecutive tokens
	// *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
	char* str_buffer = calloc((t->max_token_length*2 +1 +2), sizeof(char));
	size_t str_len = 0;

	int size = 0;



	//let's add PAD (=0), BOS (=1) and EOS (=2) entries, if required
	if(t->pad_id < 0)	{mylog(LOG_INFO,"PAD!!!"); t->pad_id = add_vocab_entry(t,"<pad>",&size); if(t->pad_id < 0)	exit(-1);}
	if(t->bos_id < 0)	{mylog(LOG_INFO,"BOS!!!"); t->bos_id = add_vocab_entry(t,"<bos>",&size); if(t->bos_id < 0)	exit(-1);}
	if(t->eos_id < 0)	{mylog(LOG_INFO,"EOS!!!"); t->eos_id = add_vocab_entry(t,"<eos>",&size); if(t->eos_id < 0)	exit(-1);}

	t->byte_fallback_offset = size;
	for(x=0;x<256;x++){
		char buf[8];
		sprintf(buf,"<0x%02X>",x);
		add_vocab_entry(t,&buf[0],&size);
	}


	// UTF-8 reference from Wikipedia:
	// Code point ↔ UTF-8 conversion
	// First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
	// U+0000	U+007F	    0xxxxxxx
	// U+0080	U+07FF	    110xxxxx	10xxxxxx
	// U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
	// U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

	// process the raw (UTF-8) byte sequence of the input string
	//
	// 1) we add basic codepoints to vocabulary
	// 2) we perform initial tokenization of the text; this will allow vocabulary building below
	//

	int ntok = 0;
	int * token = (int *)calloc((MAX_INPUT_TOKENS+42), sizeof(int));

	FILE * f;
	f = fopen(input_path,"r");
	if(f==NULL){

		mylog(LOG_ERROR, "Invalid path \"%s\" for input text",input_path); 
		exit(EXIT_FAILURE); 

	}

char * p;
char text[4096];
while((p=fgets(text,(4096-1),f))!=NULL){

	text[4095]='\0';

	for(char * c = text; *c != '\0'; c++) {

        	// reset buffer if the current byte is ASCII or a leading byte
	        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        	// 0x80 is 10000000
	        // in UTF-8, all continuation bytes start with "10" in first two bits
        	// so in English this is: "if this byte is not a continuation byte"
	        if ((*c & 0xC0) != 0x80) {
        	    // this byte must be either a leading byte (11...) or an ASCII char (0x...)
	            // => reset our location, as we're starting a new UTF-8 codepoint
        	    str_len = 0;
	        }

	        // append the current byte to the buffer
        	str_buffer[str_len++] = *c;
	        str_buffer[str_len] = '\0';

	        // while the next character is a continuation byte, continue appending
        	// but if there are too many of them, just stop to avoid overruning str_buffer size.
	        if(((*(c+1)&0xC0)==0x80) && (str_len<4)){
        	    continue;
	        }

	        // ok c+1 is not a continuation byte, so we've read in a full codepoint
		for(x=0; x<size; x++){
			if(strcmp(t->vocab[x],str_buffer)==0)	break;
		}

        	if(x==size){	//if the codepoint was not yet in the base vocabulary, try to add it
			if(add_vocab_entry(t,str_buffer,&size) < 0){
				mylog(LOG_ERROR, "vocab_size not enough to store basic codepoints"); exit(EXIT_FAILURE); 
			}
		}

		token[ntok++] = x;
        
		str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
        }
}

fclose(f);

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    int goon = 1;
    while(goon==1){

	goon = 0;

        for(x=0; x<(ntok-1); x++) {

	    if(is_separator(t->vocab[token[x  ]]) > 0) continue;
	    if(is_separator(t->vocab[token[x+1]]) > 0) continue;


            // check if we can create a new token by merging the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[token[x]], t->vocab[token[x+1]]);

	    int y;

	    for(y=0; y<size; y++){
		if(strcmp(t->vocab[y],str_buffer)==0){
			t->vocab_scores[y] += (float)(strlen(str_buffer));
			goon = 1;
			break;
		}
	    }

            if(y==size){
		
		if(add_vocab_entry(t,str_buffer,&size) < 0){

			//TODO: if vocabulary space is full, we could implement a policy for dropping lowest-score tokens;

			goon = -1;
		}
		else{
			goon = 1;
		}
            }

	    
	    if(goon==1){
		
		token[x] = y;

        	// delete token at position x+1, shift the entire sequence back 1
        	for(int i = (x+1); i<(ntok-1); i++) {
            		token[i] = token[i+1];
		}

		ntok--;
	    }
		
        }

    }

    t->vocab_size = size;

    free(str_buffer);
    free(token);
}




// ============================================================
//  Vision preprocessing
// ============================================================

byte * picture2streams(Model * model, Picture * picture){


	size_t target_picsize;
	size_t dim_stream         = model->config.dim_stream;
	size_t n_channels         = 3;	//we are working with RGB
	size_t n_patches          = (size_t)model->config.vision_image_tokens;
	size_t patch_size         = (size_t)model->config.vision_patch_size;
	size_t n_patches_per_row  = sqrt(n_patches);
	size_t patch_channel_dim  = patch_size * patch_size;


	//1) resize picture to model picture size
	if(model->config.architectures == ARCH_PALIGEMMA_CONDITIONAL){
		target_picsize = n_patches_per_row * patch_size;
		mylog(LOG_INFO, "picture resized to %d x %d pixels", target_picsize, target_picsize);
	}
	else{
		mylog(LOG_ERROR, "unknown picture target size for \"s\" vision model architecture. Exiting...", arch_text[model->config.architectures]);
		exit(1);
	}

	
	Pixel * tempix;
	tempix = (Pixel *)resize_rgb_lanczos((unsigned char *)picture->pic, picture->width, picture->height, target_picsize, target_picsize);
	free(picture->pic);
	picture->pic = tempix;
	picture->width = target_picsize;
	picture->height = target_picsize;


	//2) flatten the picture to N patches;
	//   VERSION 1 (not used):	each patch is the concatenation of the flattening of 3 squares of pixels: 
	//   				one is the R channel, one is the G channel, and one is the B channel
	//   VERSION 2 (current):	each patch is the concatenation of its pixels, and each pixel is the concatenation of R, G, and B channel values


	float ** vision_patch = malloc(sizeof(float *) * n_patches);
	for(size_t x = 0; x < n_patches ; x++){
		vision_patch[x] = malloc(n_channels * patch_size * patch_size * sizeof(float));
	}

//float IMAGENET_STANDARD_MEAN[3] = {0.485, 0.456, 0.406};
//float IMAGENET_STANDARD_STDV[3] = {0.229, 0.224, 0.225};
float IMAGENET_STANDARD_MEAN[3] = {0.5, 0.5, 0.5};
float IMAGENET_STANDARD_STDV[3] = {0.5, 0.5, 0.5};



	for(size_t row = 0; row < target_picsize; row++){

		for(size_t col = 0; col < target_picsize; col++){

			size_t x  = ((row/patch_size)*n_patches_per_row) + (col/patch_size);	//which patch
			size_t i  = (row*target_picsize) + col;					//global index of pixel
			size_t xi = ((row%patch_size)*patch_size) + (col%patch_size);		//index of pixel (its float version) within this patch (V2) or within a channel of this patch

			for(size_t channel = 0; channel < n_channels; channel++){

				float val;

				val = (float)(((byte *)&(picture->pic[i]))[channel]);


				val /= 255.0;	//range 0:255 -> 0.0:+1.0
				val -= IMAGENET_STANDARD_MEAN[channel];	//adjust to zero mean
				val /= IMAGENET_STANDARD_STDV[channel];	//adjust to std deviation 1.0

				vision_patch[x][(patch_channel_dim*channel)+xi]  = val;		//VERSION 1: one square sub-patch per channel
				//vision_patch[x][(xi*n_channels) + channel]  = val;		//VERSION 2: each patch has got interleaved channels
			}
		}
	}

		/*	
				mylog(LOG_INFO,"");
				mylog(LOG_INFO,"    USING CHANNEL-FIRST LAYOUT!   [confirmed!]");
				mylog(LOG_INFO,"");
				sleep(2);
				mylog(LOG_INFO,"");
				mylog(LOG_INFO,"    NOT USING q-k matrixes matmulf_interleaved in ENCODER (since it's not POSE_ROPE)  [confirmed!] ");
				//mylog(LOG_INFO,"    USING q-k matrixes matmulf_interleaved ALSO for POSE_LEARNED!   ");
				mylog(LOG_INFO,"");
				sleep(2);
		*/


#ifdef TRIP_DEBUG
wdebug(&vision_patch[0][0],WTYPE_FLOAT32,10,"first patch first 12 values",-1,0);
#endif



/*
	//////////////////    V1 TEST CODE: displays one picture for each channel of each patch!! (a lot of pictures)
	
	Picture * pic2 = malloc(sizeof(Picture));
	for(size_t x = 0; x < n_patches ; x++){
		for(size_t channel = 0; channel < n_channels; channel++){

			pic2->pic = calloc((patch_size*patch_size), sizeof(Pixel));
			
			for(size_t i = 0; i < (patch_size*patch_size); i++){

				((byte *)&(pic2->pic[i]))[channel] = (byte)(vision_patch[x][(patch_channel_dim*channel) + i]);
			}

			pic2->width  = patch_size;
			pic2->height = patch_size;
			displayPicture(pic2);

			free(pic2->pic);
		}
	}
	free(pic2);

	////////////////// END of TEST CODE
*/
/*
	//////////////////    V2 TEST CODE: displays one picture for each patch!! (a half-lot of pictures)
	
	Picture * pic2 = malloc(sizeof(Picture));
	pic2->pic = calloc((patch_size*patch_size)*n_patches, sizeof(Pixel));
	for(size_t x = 0; x < n_patches ; x++){


		for(size_t i = 0; i < (patch_size*patch_size); i++){

			for(size_t channel = 0; channel < n_channels; channel++){

				((byte *)&(pic2->pic[(((x/n_patches_per_row)*patch_size)+(i/patch_size))*(n_patches_per_row*patch_size) + (((x%n_patches_per_row)*patch_size) + (i%patch_size))]))[channel] = (byte)(vision_patch[x][(i*n_channels)+channel]);
			}

		}

	}
	pic2->width  = patch_size*n_patches_per_row;
	pic2->height = patch_size*n_patches_per_row;
	displayPicture(pic2);

	free(pic2->pic);
	free(pic2);

	////////////////// END of TEST CODE
*/

	//3) now we project each patch to the encoder model input size (i.e.:stream size)

	size_t flattened_patch_size = (n_channels * patch_channel_dim);


	byte * out_streams = (byte *)malloc(n_patches * dim_stream * sizeof(float));

	//convolution
	for(size_t x = 0; x < n_patches ; x++){

	    size_t x_offset = x * dim_stream * sizeof(float);

	    matmulf_nt(&model->w.vision_embeddings_w[0], (byte *)&vision_patch[x][0], flattened_patch_size, dim_stream, flattened_patch_size, 1, &out_streams[x_offset]);

	    //bias
	    sum_vectors(&out_streams[x_offset], WTYPE_FLOAT32, &model->w.vision_embeddings_b[0], wtype, dim_stream, &out_streams[x_offset]);

	}


	////posizional embeddings
	//apply_positional_embeddings(model, &out_streams[0], &out_streams[0], dim_stream, 0, 1, n_patches);



	for(size_t x = 0; x < n_patches ; x++){
		free(vision_patch[x]);
	}
	free(vision_patch);


	return out_streams;
}





// ============================================================
//  Memory management
// ============================================================

int alloc_forward_memory(Model * model, size_t B, size_t T, int action){

	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t dim_stream = (size_t)(model->config.dim_stream);
	size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
	size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
	size_t vocab_size = (size_t)(model->config.vocab_size);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_queries  = (size_t)(model->config.n_queries);
	//size_t n_keys     = (size_t)(model->config.n_keys);

	int ffn_gating;
	if((model->config.ffn_nl_type[1] == GATE_ON)  ||  (model->config.ffn_nl_type[0] == NL_SILU_LLAMA))	ffn_gating = 1;
	else													ffn_gating = 0;





	model->fm.residualstream_layerstart		= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.norm_pre_stream			= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.queries				= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.raw_attention_scores			= (byte **)myalloc(sizeof(byte *) * n_layers);
	model->fm.attention_scores			= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.heads_output				= (byte **)myalloc(sizeof(byte *) * n_layers);
	model->fm.attentionlayer_out_stream		= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.residualstream_after_attention	= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.norm_post_stream			= (byte **)myalloc(sizeof(byte *) * n_layers);
	model->fm.ffn_in_stream				= (byte **)myalloc(sizeof(byte *) * n_layers);
	model->fm.ffn_out_stream			= (byte **)myalloc(sizeof(byte *) * n_layers);
	model->fm.ffn_final_stream			= (byte **)myalloc(sizeof(byte *) * n_layers);

	model->fm.residualstream_after_ffn		= (byte **)myalloc(sizeof(byte *) * n_layers);

	if(ffn_gating==1)	model->fm.ffn_aux_stream = (byte **)myalloc(sizeof(byte *) * n_layers);
	else			model->fm.ffn_aux_stream = NULL;




	model->fm.logits	= (byte *)myalloc(vocab_size * sizeof(float) * (calculate_loss  ?  (T * B)  :  B ));	
				//if we do not need to calculate loss, only the final token of the parallelizable part of the sequence actually requires logits




	if(action == ACTION_TRAIN){

		model->fm.norm_pre_mean			= (byte **)myalloc(sizeof(byte *) * n_layers);
		model->fm.norm_pre_rstd			= (byte **)myalloc(sizeof(byte *) * n_layers);
		model->fm.norm_pre_rrms			= (byte **)myalloc(sizeof(byte *) * n_layers);


		model->fm.norm_post_mean		= (byte **)myalloc(sizeof(byte *) * n_layers);
		model->fm.norm_post_rstd		= (byte **)myalloc(sizeof(byte *) * n_layers);
		model->fm.norm_post_rrms		= (byte **)myalloc(sizeof(byte *) * n_layers);


		model->fm.norm_final_mean		= (byte *)myalloc(             sizeof(float) * T * B);
		model->fm.norm_final_rstd		= (byte *)myalloc(             sizeof(float) * T * B);
		model->fm.norm_final_rrms		= (byte *)myalloc(             sizeof(float) * T * B);

		model->fm.norm_final_stream	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);


		for(size_t layer = 0; layer < n_layers; layer++){


			if(layer == 0){
				model->fm.residualstream_layerstart[layer]	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
			}
			else{
				model->fm.residualstream_layerstart[layer]      = model->fm.residualstream_after_ffn[layer - 1];
			}

			model->fm.norm_pre_mean[layer]			= (byte *)myalloc(             sizeof(float) * T * B);
			model->fm.norm_pre_rstd[layer]			= (byte *)myalloc(             sizeof(float) * T * B);
			model->fm.norm_pre_rrms[layer]			= (byte *)myalloc(             sizeof(float) * T * B);

			model->fm.norm_pre_stream[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);

			model->fm.queries[layer]			= (byte *)myalloc(n_queries * dim_qkv * sizeof(float) * T * B);

			model->fm.raw_attention_scores[layer]		= (byte *)myalloc(n_queries * sizeof(float) * T * T * B);
			model->fm.attention_scores[layer]		= (byte *)myalloc(n_queries * sizeof(float) * T * T * B);

			model->fm.heads_output[layer]			= (byte *)myalloc(dim_stream * sizeof(float) * T * B);

			model->fm.attentionlayer_out_stream[layer]	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);

			model->fm.residualstream_after_attention[layer] 	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);

			model->fm.norm_post_mean[layer]			= (byte *)myalloc(             sizeof(float) * T * B);
			model->fm.norm_post_rstd[layer]			= (byte *)myalloc(             sizeof(float) * T * B);
			model->fm.norm_post_rrms[layer]			= (byte *)myalloc(             sizeof(float) * T * B);

			model->fm.norm_post_stream[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
			model->fm.ffn_in_stream[layer]			= (byte *)myalloc(hidden_dim * sizeof(float) * T * B);
			model->fm.ffn_out_stream[layer]			= (byte *)myalloc(hidden_dim * sizeof(float) * T * B);
			model->fm.ffn_final_stream[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);

			if(ffn_gating==1)	model->fm.ffn_aux_stream[layer]	= (byte *)myalloc(hidden_dim * sizeof(float) * T * B);
			//else			model->fm.ffn_aux_stream[layer]	= NULL;


			model->fm.residualstream_after_ffn[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
		}
		
	}
	else{


		model->fm.norm_pre_mean			= NULL;
		model->fm.norm_pre_rstd			= NULL;
		model->fm.norm_pre_rrms			= NULL;


		model->fm.norm_post_mean		= NULL;
		model->fm.norm_post_rstd		= NULL;
		model->fm.norm_post_rrms		= NULL;



		for(size_t layer = 0; layer < n_layers; layer++){


			if(layer == 0){
				model->fm.residualstream_layerstart[layer]	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
				model->fm.norm_pre_stream[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
				model->fm.queries[layer]			= (byte *)myalloc(n_queries * dim_qkv * sizeof(float) * T * B);
				model->fm.raw_attention_scores[layer]		= (byte *)myalloc(n_queries * sizeof(float) * max_tokens * max_tokens * B);
				model->fm.heads_output[layer]			= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
				model->fm.ffn_in_stream[layer]			= (byte *)myalloc(hidden_dim * sizeof(float) * T * B);
				if(ffn_gating==1)	model->fm.ffn_aux_stream[layer]	= (byte *)myalloc(hidden_dim * sizeof(float) * T * B);
				//else			model->fm.ffn_aux_stream[layer]	= NULL;
		}
			else{
				model->fm.residualstream_layerstart[layer]      = model->fm.residualstream_layerstart[0];
				model->fm.norm_pre_stream[layer]		= model->fm.norm_pre_stream[0];
				model->fm.queries[layer]			= model->fm.queries[0];
				model->fm.raw_attention_scores[layer]		= model->fm.raw_attention_scores[0];
				model->fm.heads_output[layer]			= model->fm.heads_output[0];
				model->fm.ffn_in_stream[layer]			= model->fm.ffn_in_stream[0];
				if(ffn_gating==1)	model->fm.ffn_aux_stream[layer]	= model->fm.ffn_aux_stream[0];
				//else			model->fm.ffn_aux_stream[layer]	= NULL;
			}


/*
			model->fm.norm_pre_mean[layer]			= NULL;
			model->fm.norm_pre_rstd[layer]			= NULL;
			model->fm.norm_pre_rrms[layer]			= NULL;

			model->fm.norm_post_mean[layer]			= NULL;
			model->fm.norm_post_rstd[layer]			= NULL;
			model->fm.norm_post_rrms[layer]			= NULL;
*/



			//The following activations can be efficently stored in previously used space

			//model->fm.attention_scores[layer]			= (byte *)myalloc(n_queries * sizeof(float) * T * T * B);
			//model->fm.attentionlayer_out_stream[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
			//model->fm.residualstream_after_attention[layer] 	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
			//model->fm.norm_post_stream[layer]			= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
			//model->fm.ffn_out_stream[layer]			= (byte *)myalloc(hidden_dim * sizeof(float) * T * B);
			//model->fm.ffn_final_stream[layer]			= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
			//model->fm.residualstream_after_ffn[layer]		= (byte *)myalloc(dim_stream * sizeof(float) * T * B);


			model->fm.attention_scores[layer]		= model->fm.raw_attention_scores[0];
			model->fm.attentionlayer_out_stream[layer]	= model->fm.norm_pre_stream[0];
			model->fm.residualstream_after_attention[layer]	= model->fm.residualstream_layerstart[0];
			model->fm.norm_post_stream[layer]		= model->fm.norm_pre_stream[0];
			model->fm.ffn_out_stream[layer]			= model->fm.ffn_in_stream[0];
			model->fm.ffn_final_stream[layer]		= model->fm.norm_pre_stream[0];
			model->fm.residualstream_after_ffn[layer]	= model->fm.residualstream_layerstart[0];

		}


		model->fm.norm_final_mean		= NULL;	//not layered
		model->fm.norm_final_rstd		= NULL;	//not layered
		model->fm.norm_final_rrms		= NULL;	//not layered


		//model->fm.norm_final_stream	= (byte *)myalloc(dim_stream * sizeof(float) * T * B);
		model->fm.norm_final_stream   = model->fm.residualstream_layerstart[0];

		
	}

	return 1;
}


void free_forward_memory(Model * model){
	
	size_t n_layers = model->config.n_layers;

	free_mem(model, model->fm.logits, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.norm_final_stream, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_final_mean, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_final_rstd, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_final_rrms, -1, MEMORY_FORWARD);


	for(size_t layer = 0; layer < n_layers; layer++){

		free_mem(model, model->fm.residualstream_layerstart, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.norm_pre_mean, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.norm_pre_rstd, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.norm_pre_rrms, layer, MEMORY_FORWARD);


		free_mem(model, model->fm.norm_pre_stream, layer, MEMORY_FORWARD);


		free_mem(model, model->fm.queries, layer, MEMORY_FORWARD);


		free_mem(model, model->fm.raw_attention_scores, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.attention_scores, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.heads_output, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.attentionlayer_out_stream, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.residualstream_after_attention, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.norm_post_mean, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.norm_post_rstd, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.norm_post_rrms, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.norm_post_stream, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.ffn_in_stream, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.ffn_aux_stream, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.ffn_out_stream, layer, MEMORY_FORWARD);
		free_mem(model, model->fm.ffn_final_stream, layer, MEMORY_FORWARD);

		free_mem(model, model->fm.residualstream_after_ffn, layer, MEMORY_FORWARD);

	}

	free_mem(model, model->fm.residualstream_layerstart, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.norm_pre_mean, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_pre_rstd, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_pre_rrms, -1, MEMORY_FORWARD);


	free_mem(model, model->fm.norm_pre_stream, -1, MEMORY_FORWARD);


	free_mem(model, model->fm.queries, -1, MEMORY_FORWARD);


	free_mem(model, model->fm.raw_attention_scores, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.attention_scores, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.heads_output, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.attentionlayer_out_stream, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.residualstream_after_attention, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.norm_post_mean, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_post_rstd, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.norm_post_rrms, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.norm_post_stream, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.ffn_in_stream, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.ffn_aux_stream, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.ffn_out_stream, -1, MEMORY_FORWARD);
	free_mem(model, model->fm.ffn_final_stream, -1, MEMORY_FORWARD);

	free_mem(model, model->fm.residualstream_after_ffn, -1, MEMORY_FORWARD);
}



int free_mem(Model * model, void * addr, ssize_t layer, int memory){
	
   size_t n_layers = model->config.n_layers;


   if(addr == NULL){
#ifdef TRIP_DEBUG
	mylog(LOG_DEBUG,"free_mem NULL: skipped");
#endif
	return -1;
   }

   if(layer != -1){
	addr = (void *)(((byte **)addr)[layer]);
   }

   free(addr);


   if(memory == MEMORY_FORWARD){

#ifdef TRIP_DEBUG
	mylog(LOG_DEBUG,"free_mem FORWARD");
#endif


	//non-layered
	if(addr == model->fm.logits)			model->fm.logits = NULL;

	if(addr == model->fm.norm_final_stream)		model->fm.norm_final_stream = NULL;

	if(addr == model->fm.norm_final_mean)		model->fm.norm_final_mean = NULL;
	if(addr == model->fm.norm_final_rstd)		model->fm.norm_final_rstd = NULL;
	if(addr == model->fm.norm_final_rrms)		model->fm.norm_final_rrms = NULL;


	//layered
	if(addr == model->fm.residualstream_layerstart)		model->fm.residualstream_layerstart = NULL;

	if(addr == model->fm.norm_pre_mean)			model->fm.norm_pre_mean = NULL;
	if(addr == model->fm.norm_pre_rstd)			model->fm.norm_pre_rstd = NULL;
	if(addr == model->fm.norm_pre_rrms)			model->fm.norm_pre_rrms = NULL;


	if(addr == model->fm.norm_pre_stream)			model->fm.norm_pre_stream = NULL;


	if(addr == model->fm.queries)				model->fm.queries = NULL;


	if(addr == model->fm.raw_attention_scores)		model->fm.raw_attention_scores = NULL;
	if(addr == model->fm.attention_scores)			model->fm.attention_scores = NULL;

	if(addr == model->fm.heads_output)			model->fm.heads_output = NULL;

	if(addr == model->fm.attentionlayer_out_stream)		model->fm.attentionlayer_out_stream = NULL;

	if(addr == model->fm.residualstream_after_attention)	model->fm.residualstream_after_attention = NULL;

	if(addr == model->fm.norm_post_mean)			model->fm.norm_post_mean = NULL;
	if(addr == model->fm.norm_post_rstd)			model->fm.norm_post_rstd = NULL;
	if(addr == model->fm.norm_post_rrms)			model->fm.norm_post_rrms = NULL;

	if(addr == model->fm.norm_post_stream)			model->fm.norm_post_stream = NULL;

	if(addr == model->fm.ffn_in_stream)			model->fm.ffn_in_stream = NULL;
	if(addr == model->fm.ffn_aux_stream)			model->fm.ffn_aux_stream = NULL;
	if(addr == model->fm.ffn_out_stream)			model->fm.ffn_out_stream = NULL;
	if(addr == model->fm.ffn_final_stream)			model->fm.ffn_final_stream = NULL;

	if(addr == model->fm.residualstream_after_ffn)		model->fm.residualstream_after_ffn = NULL;


	for(size_t layer = 0; layer < n_layers; layer++){

#ifdef TRIP_DEBUG
		mylog(LOG_DEBUG,"free_mem FORWARD layer %d", layer);
#endif

		if((model->fm.residualstream_layerstart != NULL)  &&  (addr == model->fm.residualstream_layerstart[layer]))		model->fm.residualstream_layerstart[layer] = NULL;

		if((model->fm.norm_pre_mean != NULL)  &&  (addr == model->fm.norm_pre_mean[layer]))			model->fm.norm_pre_mean[layer] = NULL;
		if((model->fm.norm_pre_rstd != NULL)  &&  (addr == model->fm.norm_pre_rstd[layer]))			model->fm.norm_pre_rstd[layer] = NULL;
		if((model->fm.norm_pre_rrms != NULL)  &&  (addr == model->fm.norm_pre_rrms[layer]))			model->fm.norm_pre_rrms[layer] = NULL;


		if((model->fm.norm_pre_stream != NULL)  &&  (addr == model->fm.norm_pre_stream[layer]))			model->fm.norm_pre_stream[layer] = NULL;


		if((model->fm.queries != NULL)  &&  (addr == model->fm.queries[layer]))				model->fm.queries[layer] = NULL;


		if((model->fm.raw_attention_scores != NULL)  &&  (addr == model->fm.raw_attention_scores[layer]))		model->fm.raw_attention_scores[layer] = NULL;
		if((model->fm.attention_scores != NULL)  &&  (addr == model->fm.attention_scores[layer]))			model->fm.attention_scores[layer] = NULL;

		if((model->fm.heads_output != NULL)  &&  (addr == model->fm.heads_output[layer]))			model->fm.heads_output[layer] = NULL;

		if((model->fm.attentionlayer_out_stream != NULL)  &&  (addr == model->fm.attentionlayer_out_stream[layer]))		model->fm.attentionlayer_out_stream[layer] = NULL;

		if((model->fm.residualstream_after_attention != NULL)  &&  (addr == model->fm.residualstream_after_attention[layer]))	model->fm.residualstream_after_attention[layer] = NULL;

		if((model->fm.norm_post_mean != NULL)  &&  (addr == model->fm.norm_post_mean[layer]))			model->fm.norm_post_mean[layer] = NULL;
		if((model->fm.norm_post_rstd != NULL)  &&  (addr == model->fm.norm_post_rstd[layer]))			model->fm.norm_post_rstd[layer] = NULL;
		if((model->fm.norm_post_rrms != NULL)  &&  (addr == model->fm.norm_post_rrms[layer]))			model->fm.norm_post_rrms[layer] = NULL;

		if((model->fm.norm_post_stream != NULL)  &&  (addr == model->fm.norm_post_stream[layer]))			model->fm.norm_post_stream[layer] = NULL;

		if((model->fm.ffn_in_stream != NULL)  &&  (addr == model->fm.ffn_in_stream[layer]))			model->fm.ffn_in_stream[layer] = NULL;
		if((model->fm.ffn_aux_stream != NULL)  &&  (addr == model->fm.ffn_aux_stream[layer]))			model->fm.ffn_aux_stream[layer] = NULL;
		if((model->fm.ffn_out_stream != NULL)  &&  (addr == model->fm.ffn_out_stream[layer]))			model->fm.ffn_out_stream[layer] = NULL;
		if((model->fm.ffn_final_stream != NULL)  &&  (addr == model->fm.ffn_final_stream[layer]))			model->fm.ffn_final_stream[layer] = NULL;

		if((model->fm.residualstream_after_ffn != NULL)  &&  (addr == model->fm.residualstream_after_ffn[layer]))		model->fm.residualstream_after_ffn[layer] = NULL;

	}
	
   }
   else
   if(memory == MEMORY_GRADIENTS){

	//non-layered
	if(addr == model->grads.dlogits)			model->grads.dlogits = NULL;
	if(addr == model->grads.dlogits_classifier)		model->grads.dlogits_classifier = NULL;
	if(addr == model->grads.dnorm_final_stream)		model->grads.dnorm_final_stream = NULL;
	if(addr == model->grads.dnorm_final_w)			model->grads.dnorm_final_w = NULL;
	if(addr == model->grads.dnorm_final_b)			model->grads.dnorm_final_b = NULL;


	//layered
	if(addr == model->grads.dresidualstream_after_ffn)		model->grads.dresidualstream_after_ffn = NULL;

	if(addr == model->grads.dffn_final_stream)			model->grads.dffn_final_stream = NULL;
	if(addr == model->grads.dffn_out_stream)			model->grads.dffn_out_stream = NULL;
	if(addr == model->grads.dffn_aux_stream)			model->grads.dffn_aux_stream = NULL;
	if(addr == model->grads.dffn_in_stream)				model->grads.dffn_in_stream = NULL;

	if(addr == model->grads.dpost_ffn_w)				model->grads.dpost_ffn_w = NULL;
	if(addr == model->grads.dpost_ffn_b)				model->grads.dpost_ffn_b = NULL;

	if(addr == model->grads.dnorm_post_stream)			model->grads.dnorm_post_stream = NULL;

	if(addr == model->grads.dnorm_post_w)				model->grads.dnorm_post_w = NULL;
	if(addr == model->grads.dnorm_post_b)				model->grads.dnorm_post_b = NULL;

	if(addr == model->grads.dresidualstream_after_attention)	model->grads.dresidualstream_after_attention = NULL;

	if(addr == model->grads.dattentionlayer_out_stream)		model->grads.dattentionlayer_out_stream = NULL;

	if(addr == model->grads.dom)					model->grads.dom = NULL;
	if(addr == model->grads.dob)					model->grads.dob = NULL;


	if(addr == model->grads.dheads_output)				model->grads.dheads_output = NULL;

	if(addr == model->grads.dattention_scores)			model->grads.dattention_scores = NULL;
	if(addr == model->grads.draw_attention_scores)			model->grads.draw_attention_scores = NULL;

	if(addr == model->grads.dqueries)				model->grads.dqueries = NULL;
	if(addr == model->grads.dkeys)					model->grads.dkeys = NULL;
	if(addr == model->grads.dvalues)				model->grads.dvalues = NULL;

	if(addr == model->grads.dqm)					model->grads.dqm = NULL;
	if(addr == model->grads.dqb)					model->grads.dqb = NULL;
	if(addr == model->grads.dkm)					model->grads.dkm = NULL;
	if(addr == model->grads.dkb)					model->grads.dkb = NULL;
	if(addr == model->grads.dvm)					model->grads.dvm = NULL;
	if(addr == model->grads.dvb)					model->grads.dvb = NULL;

	if(addr == model->grads.dnorm_pre_stream)			model->grads.dnorm_pre_stream = NULL;

	if(addr == model->grads.dnorm_pre_w)				model->grads.dnorm_pre_w = NULL;
	if(addr == model->grads.dnorm_pre_b)				model->grads.dnorm_pre_b = NULL;


	if(addr == model->grads.dresidualstream_layerstart)		model->grads.dresidualstream_layerstart = NULL;




	for(size_t layer = 0; layer < n_layers; layer++){

		if((model->grads.dresidualstream_after_ffn != NULL)  &&  (addr == model->grads.dresidualstream_after_ffn[layer]))	model->grads.dresidualstream_after_ffn[layer] = NULL;

		if((model->grads.dffn_final_stream != NULL)  &&  (addr == model->grads.dffn_final_stream[layer]))		model->grads.dffn_final_stream[layer] = NULL;
		if((model->grads.dffn_out_stream != NULL)  &&  (addr == model->grads.dffn_out_stream[layer]))			model->grads.dffn_out_stream[layer] = NULL;
		if((model->grads.dffn_aux_stream != NULL)  &&  (addr == model->grads.dffn_aux_stream[layer]))			model->grads.dffn_aux_stream[layer] = NULL;
		if((model->grads.dffn_in_stream != NULL)  &&  (addr == model->grads.dffn_in_stream[layer]))			model->grads.dffn_in_stream[layer] = NULL;

		if((model->grads.dpost_ffn_w != NULL)  &&  (addr == model->grads.dpost_ffn_w[layer]))			model->grads.dpost_ffn_w[layer] = NULL;
		if((model->grads.dpost_ffn_b != NULL)  &&  (addr == model->grads.dpost_ffn_b[layer]))			model->grads.dpost_ffn_b[layer] = NULL;

		if((model->grads.dnorm_post_stream != NULL)  &&  (addr == model->grads.dnorm_post_stream[layer]))		model->grads.dnorm_post_stream[layer] = NULL;
		
		if((model->grads.dnorm_post_w != NULL)  &&  (addr == model->grads.dnorm_post_w[layer]))			model->grads.dnorm_post_w[layer] = NULL;
		if((model->grads.dnorm_post_b != NULL)  &&  (addr == model->grads.dnorm_post_b[layer]))			model->grads.dnorm_post_b[layer] = NULL;

		if((model->grads.dresidualstream_after_attention != NULL)  &&  (addr == model->grads.dresidualstream_after_attention[layer]))	model->grads.dresidualstream_after_attention[layer] = NULL;
		
		if((model->grads.dattentionlayer_out_stream != NULL)  &&  (addr == model->grads.dattentionlayer_out_stream[layer]))	model->grads.dattentionlayer_out_stream[layer] = NULL;

		if((model->grads.dom != NULL)  &&  (addr == model->grads.dom[layer]))				model->grads.dom[layer] = NULL;
		if((model->grads.dob != NULL)  &&  (addr == model->grads.dob[layer]))				model->grads.dob[layer] = NULL;

		if((model->grads.dheads_output != NULL)  &&  (addr == model->grads.dheads_output[layer]))			model->grads.dheads_output[layer] = NULL;

		if((model->grads.draw_attention_scores != NULL)  &&  (addr == model->grads.draw_attention_scores[layer]))		model->grads.draw_attention_scores[layer] = NULL;
		if((model->grads.dattention_scores != NULL)  &&  (addr == model->grads.dattention_scores[layer]))		model->grads.dattention_scores[layer] = NULL;


		if((model->grads.dqueries != NULL)  &&  (addr == model->grads.dqueries[layer]))			model->grads.dqueries[layer] = NULL;
		if((model->grads.dkeys != NULL)  &&  (addr == model->grads.dkeys[layer]))				model->grads.dkeys[layer] = NULL;
		if((model->grads.dvalues != NULL)  &&  (addr == model->grads.dvalues[layer]))				model->grads.dvalues[layer] = NULL;

		if((model->grads.dqm != NULL)  &&  (addr == model->grads.dqm[layer]))				model->grads.dqm[layer] = NULL;
		if((model->grads.dqb != NULL)  &&  (addr == model->grads.dqb[layer]))				model->grads.dqb[layer] = NULL;
		if((model->grads.dkm != NULL)  &&  (addr == model->grads.dkm[layer]))				model->grads.dkm[layer] = NULL;
		if((model->grads.dkb != NULL)  &&  (addr == model->grads.dkb[layer]))				model->grads.dkb[layer] = NULL;
		if((model->grads.dvm != NULL)  &&  (addr == model->grads.dvm[layer]))				model->grads.dvm[layer] = NULL;
		if((model->grads.dvb != NULL)  &&  (addr == model->grads.dvb[layer]))				model->grads.dvb[layer] = NULL;




		if((model->grads.dnorm_pre_stream != NULL)  &&  (addr == model->grads.dnorm_pre_stream[layer]))		model->grads.dnorm_pre_stream[layer] = NULL;

		if((model->grads.dnorm_pre_w != NULL)  &&  (addr == model->grads.dnorm_pre_w[layer]))			model->grads.dnorm_pre_w[layer] = NULL;
		if((model->grads.dnorm_pre_b != NULL)  &&  (addr == model->grads.dnorm_pre_b[layer]))			model->grads.dnorm_pre_b[layer] = NULL;

		if((model->grads.dresidualstream_layerstart != NULL)  &&  (addr == model->grads.dresidualstream_layerstart[layer]))	model->grads.dresidualstream_layerstart[layer] = NULL;

	}


   }


   return 1;

}



int alloc_gradients_memory(Model * model, size_t B, size_t T){	

	//if(runtime_actions & (1<<SIGUSR2))	return -1;

	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_keys     = (size_t)(model->config.n_keys);

	size_t dim_stream = (size_t)(model->config.dim_stream);
	//size_t max_tokens = (size_t)((model->config.submodel_type==MODELTYPE_DECODER)?(model->config.sequence_maxtokens):(model->config.vision_image_tokens));
	size_t hidden_dim = (size_t)(model->config.ffn_hidden_dim);
	size_t vocab_size = (size_t)(model->config.vocab_size);
	size_t n_queries  = (size_t)(model->config.n_queries);
	//size_t n_keys     = (size_t)(model->config.n_keys);



	//just for clarity
	size_t batch_size = B;
	size_t max_tokens = T;

	int ffn_gating;
	if((model->config.ffn_nl_type[1] == GATE_ON)  ||  (model->config.ffn_nl_type[0] == NL_SILU_LLAMA))	ffn_gating = 1;
	else													ffn_gating = 0;



	////let's alloc pointers to all the layers
	////model->fm.keys   = (byte **)myalloc( n_layers * sizeof(byte *));
	////model->fm.values = (byte **)myalloc( n_layers * sizeof(byte *));


	model->grads.dlogits				= (byte *)myalloc(             vocab_size * sizeof(float) * B * T );
	model->grads.dlogits_classifier			= (byte *)myalloc(dim_stream * vocab_size * sizeof(float) 	  );	//weights matrix: gradients will cumulate over B and T
	model->grads.dnorm_final_stream			= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );
	model->grads.dnorm_final_w			= (byte *)myalloc(dim_stream *              sizeof(float)	  );	//weights vector: gradients will cumulate over B and T
	model->grads.dnorm_final_b			= (byte *)myalloc(dim_stream *              sizeof(float)	  );	//weights vector: gradients will cumulate over B and T


	model->grads.dresidualstream_after_ffn		= (byte **)myalloc(n_layers * sizeof(byte *)        );


	model->grads.dffn_final_stream			= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dpost_ffn_w			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dpost_ffn_b			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	
	model->grads.dffn_out_stream			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dffn_aux_stream			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dffn_in_stream			= (byte **)myalloc(n_layers * sizeof(byte *)        );

if(ffn_gating == 1){	
	model->grads.dpre_ffn_w2			= (byte **)myalloc(n_layers * sizeof(byte *)        );
}

	model->grads.dpre_ffn_w				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dpre_ffn_b				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	

	model->grads.dnorm_post_stream			= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dnorm_post_w			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dnorm_post_b			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	
	model->grads.dresidualstream_after_attention	= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dattentionlayer_out_stream		= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dom				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dob				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dheads_output			= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dattention_scores			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.draw_attention_scores		= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dqueries				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dkeys				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dvalues				= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dqm				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dqb				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dkm				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dkb				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dvm				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dvb				= (byte **)myalloc(n_layers * sizeof(byte *)        );
	

	model->grads.dnorm_pre_stream			= (byte **)myalloc(n_layers * sizeof(byte *)        );

	model->grads.dnorm_pre_w			= (byte **)myalloc(n_layers * sizeof(byte *)        );
	model->grads.dnorm_pre_b			= (byte **)myalloc(n_layers * sizeof(byte *)        );

	
	model->grads.dresidualstream_layerstart		= (byte **)myalloc(n_layers * sizeof(byte *)        );


	for(size_t layer = 0; layer < n_layers; layer++){

		model->grads.dresidualstream_after_ffn[layer]		= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );


		model->grads.dffn_final_stream[layer]			= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );

		model->grads.dpost_ffn_w[layer]				= (byte *)myalloc(dim_stream * hidden_dim * sizeof(float)         );
		model->grads.dpost_ffn_b[layer]				= (byte *)myalloc(dim_stream *              sizeof(float)         );

		model->grads.dffn_out_stream[layer]			= (byte *)myalloc(             hidden_dim * sizeof(float) * B * T );
		model->grads.dffn_aux_stream[layer]			= (byte *)myalloc(             hidden_dim * sizeof(float) * B * T );
		model->grads.dffn_in_stream[layer]			= (byte *)myalloc(             hidden_dim * sizeof(float) * B * T );

	if(ffn_gating == 1){
		model->grads.dpre_ffn_w2[layer]				= (byte *)myalloc(dim_stream * hidden_dim * sizeof(float)         );
	}

		model->grads.dpre_ffn_w[layer]				= (byte *)myalloc(dim_stream * hidden_dim * sizeof(float)         );
		model->grads.dpre_ffn_b[layer]				= (byte *)myalloc(             hidden_dim * sizeof(float)         );


		model->grads.dnorm_post_stream[layer]			= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );

		model->grads.dnorm_post_w[layer]			= (byte *)myalloc(dim_stream *              sizeof(float)         );
		model->grads.dnorm_post_b[layer]			= (byte *)myalloc(dim_stream *              sizeof(float)         );

		model->grads.dresidualstream_after_attention[layer]	= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );

		model->grads.dattentionlayer_out_stream[layer]		= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );

		model->grads.dom[layer]					= (byte *)myalloc(dim_stream * dim_stream * sizeof(float) );
		model->grads.dob[layer]					= (byte *)myalloc(dim_stream *              sizeof(float) );

		model->grads.dheads_output[layer]			= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );
	
		model->grads.draw_attention_scores[layer]		= (byte *)myalloc(n_queries  *              sizeof(float) * B * T * T);
		model->grads.dattention_scores[layer]			= (byte *)myalloc(n_queries  *              sizeof(float) * B * T * T);

		model->grads.dqueries[layer]				= (byte *)myalloc(n_queries  * dim_qkv    * sizeof(float) * B * T );
		model->grads.dkeys[layer]				= (byte *)myalloc(n_keys     * dim_qkv    * sizeof(float) * B * T );
		model->grads.dvalues[layer]				= (byte *)myalloc(n_keys     * dim_qkv    * sizeof(float) * B * T );

		model->grads.dqm[layer]					= (byte *)myalloc(dim_stream * dim_qkv    * sizeof(float)         * n_queries );
		model->grads.dqb[layer]					= (byte *)myalloc(             dim_qkv    * sizeof(float)         * n_queries );
		model->grads.dkm[layer]					= (byte *)myalloc(dim_stream * dim_qkv    * sizeof(float)         * n_keys    );
		model->grads.dkb[layer]					= (byte *)myalloc(             dim_qkv    * sizeof(float)         * n_keys    );
		model->grads.dvm[layer]					= (byte *)myalloc(dim_stream * dim_qkv    * sizeof(float)         * n_keys    );
		model->grads.dvb[layer]					= (byte *)myalloc(             dim_qkv    * sizeof(float)         * n_keys    );


		model->grads.dnorm_pre_stream[layer]			= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );


		model->grads.dnorm_pre_w[layer]				= (byte *)myalloc(dim_stream *              sizeof(float)         );
		model->grads.dnorm_pre_b[layer]				= (byte *)myalloc(dim_stream *              sizeof(float)         );


		if(layer == 0){
			model->grads.dresidualstream_layerstart[layer]		= (byte *)myalloc(dim_stream *              sizeof(float) * B * T );
		}
		else{
			model->grads.dresidualstream_layerstart[layer]		= model->grads.dresidualstream_after_ffn[layer - 1];
		}


	}


	if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){

		model->grads.dembeddings	= (byte *)myalloc(dim_stream * vocab_size * sizeof(float) 	  );	//weights matrix: gradients will cumulate over B and T
	}
	else{
		model->grads.dembeddings        = model->grads.dlogits_classifier;
	}


	if(model->config.pose_cfg == POSE_LEARNED){

		mylog(LOG_VERBOSE_DEBUG,"");
		mylog(LOG_VERBOSE_DEBUG,"model->config.pose_cfg == POSE_LEARNED, bytes = %zd",(dim_stream * max_tokens * sizeof(float)));
		mylog(LOG_VERBOSE_DEBUG,"");
		sleep(2);

		model->grads.dlearned_pose_w	= (byte *)myalloc(dim_stream * max_tokens * sizeof(float) 	  );	//weights matrix: gradients will cumulate over B and T
	}

}


int free_gradients_memory(Model * model, size_t B, size_t T){	

	//size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_keys     = (size_t)(model->config.n_keys);

	//just for clarity
	//size_t batch_size = B;
	//size_t max_tokens = T;



	free_mem(model, model->grads.dlogits, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dlogits_classifier, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dnorm_final_stream, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dnorm_final_w, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dnorm_final_b, -1, MEMORY_GRADIENTS);

	for(size_t layer = 0; layer < n_layers; layer++){

		free_mem(model, model->grads.dresidualstream_after_ffn, layer, MEMORY_GRADIENTS);


		free_mem(model, model->grads.dpost_ffn_w, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dpost_ffn_b, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dffn_final_stream, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dffn_out_stream, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dffn_aux_stream, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dffn_in_stream, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dpre_ffn_w2, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dpre_ffn_w, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dpre_ffn_b, layer, MEMORY_GRADIENTS);


		free_mem(model, model->grads.dnorm_post_stream, layer, MEMORY_GRADIENTS);
		
		free_mem(model, model->grads.dnorm_post_w, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dnorm_post_b, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dresidualstream_after_attention, layer, MEMORY_GRADIENTS);
		
		free_mem(model, model->grads.dattentionlayer_out_stream, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dom, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dob, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dheads_output, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.draw_attention_scores, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dattention_scores, layer, MEMORY_GRADIENTS);


		free_mem(model, model->grads.dqueries, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dkeys, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dvalues, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dqm, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dqb, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dkm, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dkb, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dvm, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dvb, layer, MEMORY_GRADIENTS);




		free_mem(model, model->grads.dnorm_pre_stream, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dnorm_pre_w, layer, MEMORY_GRADIENTS);
		free_mem(model, model->grads.dnorm_pre_b, layer, MEMORY_GRADIENTS);

		free_mem(model, model->grads.dresidualstream_layerstart, layer, MEMORY_GRADIENTS);
	}


	free_mem(model, model->grads.dresidualstream_after_ffn, -1, MEMORY_GRADIENTS);


	free_mem(model, model->grads.dpost_ffn_w, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dpost_ffn_b, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dffn_final_stream, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dffn_out_stream, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dffn_aux_stream, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dffn_in_stream, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dpre_ffn_w2, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dpre_ffn_w, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dpre_ffn_b, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dnorm_post_stream, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dnorm_post_w, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dnorm_post_b, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dresidualstream_after_attention, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dattentionlayer_out_stream, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dom, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dob, -1, MEMORY_GRADIENTS);


	free_mem(model, model->grads.dheads_output, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dattention_scores, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.draw_attention_scores, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dqueries, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dkeys, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dvalues, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dqm, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dqb, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dkm, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dkb, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dvm, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dvb, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dnorm_pre_stream, -1, MEMORY_GRADIENTS);

	free_mem(model, model->grads.dnorm_pre_w, -1, MEMORY_GRADIENTS);
	free_mem(model, model->grads.dnorm_pre_b, -1, MEMORY_GRADIENTS);


	free_mem(model, model->grads.dresidualstream_layerstart, -1, MEMORY_GRADIENTS);


	if(model->config.embeddings_cfg == EMBEDDINGS_UNSHARED){

		free_mem(model, model->grads.dembeddings, -1, MEMORY_GRADIENTS);
	}

	free_mem(model, model->grads.dlearned_pose_w, -1, MEMORY_GRADIENTS);

}





int alloc_kv_memory(Model * model, size_t B, size_t T, size_t this_action){	

	//NOTE: "this_action" may not be the global action (e.g.: in ACTION_TRAIN, we will do ACTION_DECODE every N steps to evaluate the model)

	size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);
	size_t n_layers   = (size_t)(model->config.n_layers);
	size_t n_keys     = (size_t)(model->config.n_keys);

	//just for clarity
	size_t batch_size = B;
	size_t max_tokens = T;


	//let's alloc pointers to all the layers
	model->fm.keys   = (byte **)myalloc( n_layers * sizeof(byte *));
	model->fm.values = (byte **)myalloc( n_layers * sizeof(byte *));


	//even cached values, always in float32 format
	
	if( 	(this_action == ACTION_DECODE)
		||
		(this_action == ACTION_CHAT)
		||
		((this_action == ACTION_VISION) && (model->config.submodel_type==MODELTYPE_DECODER))	//in decoder, we will process tokens sequentially: full kv_cache REQUIRED
		||
		(this_action == ACTION_TRAIN)	//while it's true that in the forward pass we do not need a full kv memory, since we process all tokens in parallel, we need to keep track of the activations for the backward pass
	){

		size_t layer;

		//for all the above actions, we need to cache keys and values for all layers
		for(layer = 0; layer < n_layers; layer++){
			model->fm.keys[layer]   = (byte *)myalloc( n_keys * (dim_qkv * max_tokens * batch_size) * sizeof(float));
			model->fm.values[layer] = (byte *)myalloc( n_keys * (dim_qkv * max_tokens * batch_size) * sizeof(float));
		}
	}
	else
	if(	((this_action == ACTION_VISION) && (model->config.submodel_type==MODELTYPE_VISION_ENCODER))    //in encoder, we will process tokens in parallel: full kv_cache NOT required
){

		size_t layer;

		//for the actions above, we don't need to cache each layer's keys and values
		model->fm.keys[0]   = (byte *)myalloc( n_keys * (dim_qkv * max_tokens * batch_size) * sizeof(float));
		model->fm.values[0] = (byte *)myalloc( n_keys * (dim_qkv * max_tokens * batch_size) * sizeof(float));

		//we will just re-use the memory of the first layer for all the other layers
		for(layer = 1; layer < n_layers; layer++){
			model->fm.keys[layer]   = model->fm.keys[0];
			model->fm.values[layer] = model->fm.values[0];
		}

	}
	else{

		mylog(LOG_ERROR,"action \"%s\" not handled in kv_memory creation. To be implemented! Exiting...",action_text[this_action]);
		exit(1);
	}
}



void free_kv_memory(Model * model){

	size_t n_layers = model->config.n_layers;

	if(model->fm.keys != NULL){
		//deduplicate: if multiple layers share the same allocation, NULL the duplicates first
		for(int layer = 0; layer < n_layers; layer++){
			if(model->fm.keys[layer] != NULL){
				for(int j = layer + 1; j < n_layers; j++){
					if(model->fm.keys[j] == model->fm.keys[layer])
						model->fm.keys[j] = NULL;
				}
			}
		}
		for(int layer = 0; layer < n_layers; layer++){
			if(model->fm.keys[layer] != NULL)	free(model->fm.keys[layer]);
			model->fm.keys[layer] = NULL;
		}

		free(model->fm.keys);
		model->fm.keys = NULL;
	}
	if(model->fm.values != NULL){
		//deduplicate: if multiple layers share the same allocation, NULL the duplicates first
		for(int layer = 0; layer < n_layers; layer++){
			if(model->fm.values[layer] != NULL){
				for(int j = layer + 1; j < n_layers; j++){
					if(model->fm.values[j] == model->fm.values[layer])
						model->fm.values[j] = NULL;
				}
			}
		}
		for(int layer = 0; layer < n_layers; layer++){
			if(model->fm.values[layer] != NULL)	free(model->fm.values[layer]);
			model->fm.values[layer] = NULL;
		}

		free(model->fm.values);
		model->fm.values = NULL;
	}

}



void free_model_memory(Model * model){

	
	//free_kv_memory(model);


	if(rope_subk != NULL){
		free(rope_subk);
		rope_subk = NULL;
	}


	if(toki.vocab != NULL){
		for(int i=0; i<toki.vocab_size; i++){
			if(toki.vocab[i] != NULL){
				free(toki.vocab[i]);
				toki.vocab[i] = NULL;
			}
		}
		free(toki.vocab);
		toki.vocab = NULL;
	}
	if(toki.vocab_scores != NULL){
		free(toki.vocab_scores);
		toki.vocab_scores = NULL;
	}
	if(toki.sorted_vocab != NULL){
		free(toki.sorted_vocab);
		toki.sorted_vocab = NULL;
	}

	if(ramflag == RAM_BASIC_OPTIMIZATIONS){
		for(int i=0; i<model->nfiles; i++){
			if((model->file_data[i] != NULL)  &&  (model->file_data[i] != MAP_FAILED)){
				munmap(model->file_data[i], model->checkpoint_size[i]);
			}
			if(model->fd[i] > 0){
				close(model->fd[i]);
			}
		}
	}
	else
	if(ramflag == RAM_NO_OPTIMIZATIONS){
	    if(model->config.shared_weights_memory == false){
		for(int i=0; i<model->nfiles; i++){
			if(model->header_data[i])	free(model->header_data[i]);
			if(model->tensors_data[i])	free(model->tensors_data[i]);
			model->header_data[i]  = NULL;
			model->tensors_data[i] = NULL;
		}
	    }
	}


	if(model->config.shared_weights_memory == false){
		//let's free the space for the pointers for all the per-layer tensors
		free(model->w.norm_pre_w);
		free(model->w.norm_pre_b);
		free(model->w.qm);
		free(model->w.km);
		free(model->w.vm);
		free(model->w.om);
		free(model->w.qb);
		free(model->w.kb);
		free(model->w.vb);
		free(model->w.ob);
		free(model->w.norm_post_w);
		free(model->w.norm_post_b);
		free(model->w.pre_ffn_w);
		free(model->w.pre_ffn_b);
		free(model->w.pre_ffn_w2);
		free(model->w.post_ffn_w);
		free(model->w.post_ffn_b);
		//and the whole data spaces
		free(model->header_data);
		free(model->tensors_data);
	}


	free(model->dirpath);
	free(model->file_data);
	free(model->fd);
	free(model->checkpoint_size);

	free_forward_memory(model);	

	free(model);
}


