#define TRIP_MAIN
#include "trip.h"

// ============================================================
//  Global variable definitions
// ============================================================

char lbuf[LLEN];
char inbuf[MAX_TEXT_LEN+1];

char user_prompt[PROMPT_MAXLEN];
char vision_picture_path[LLEN];

const int wtype_bytesize [] = { 0 , 2 , 4 , 2 };
char * wtype_text [] = { "undefined" , "bf16" , "float32" , "float16" };

int wtype = WTYPE_UNDEFINED;
size_t wsize = 0;

//this is for model conversion
int target_wtype = WTYPE_UNDEFINED;
size_t target_wsize = 0;

float norm_eps = 1e-05;
float pose_theta = 10000.0;

int log_cfg = LOG_VERBOSE_DEBUG;
// Alternative: set log_cfg = LOG_INFO for quieter output

// Alternative: set calculate_loss = true to compute loss during inference
bool calculate_loss = false;

int runtime_actions = 0;

int build_vocab_size;
char * input_text_path = NULL;

char * tok_path = "tokenizer.json";	//default
char * mod_path = "model.safetensors";	//default
char * cfg_path = NULL;			//used only for SAFETENSORS model format;
					//we will build the full path when the user will provide the path of the (first?) checkpoint file



char * train_cfg_path  = "training_args.json";	//default
char * train_data_path = "training_data.txt";	//default

ssize_t training_batch_size =  1;
ssize_t training_max_steps  = -1;
ssize_t training_num_epochs =  1;
ssize_t training_save_steps = -1;

char training_lr_scheduler_type[32] = "constant";
ssize_t training_warmup_steps = 0;
float training_min_lr = 0.0;

char * chat_context_file = NULL;
int chat_flags = 0x00000000;

int vision_detect_status = 0;
int vision_detect_y_min;
int vision_detect_x_min;
int vision_detect_y_max;
int vision_detect_x_max;

const char * log_label[] = { "" , "ERROR: " , "" , "" , "" , "" , "" };

int ** input_tokens   = NULL;
int *  n_input_tokens = NULL;

int parallel_forwarding = PARALLEL_FORWARDING_ON;
int ramflag = RAM_NO_OPTIMIZATIONS;

const char * action_text[] = { "NONE" , "DECODE" , "CHAT" , "VISION" , "TRAIN" , "CREATE" , "CREATE_VOCABULARY" };
int action = ACTION_NONE;

const char * arch_text[] = { "BASIC" , "LlamaForCausalLM" , "GemmaForCausalLM" , "PaliGemmaForConditionalGeneration" , "GPT_CAUSAL" };

const char * modeltype_text[] = { "undefined" , "DECODER" , "VISION_ENCODER" };

const char * chatscheme_text[] = { "none" , "Llama(not Tiny)" , "Gemma" , "TinyLlama" };
int chat_scheme = CHATSCHEME_NONE;

const char * tokformat_text[] = { "TRIP" , "LLAMA2_AK" , "JSON_HUGGINGFACE" , "GPT2_AK" };
int tokenizer_format = TOKFORMAT_TRIP;

const char * toktype_text[] = { "TRIP" , "SENTENCEPIECE" };
int tokenizer_type = TOKTYPE_SENTENCEPIECE;

const char * cp_litteral[] = { "UNDEFINED!" , "LLAMA2_AK" , "SAFETENSORS" , "GPT2_AK" };
int checkpoint_type = CP_UNDEFINED;

const char * embeddings_cfg_text[] = { "UNSHARED", "SHARED" };

const char * pose_cfg_text[] = { "undefined", "ORIGINAL", "ROPE", "LEARNED" };


float * rope_subk = NULL;
size_t rope_subk_timesteps = 0;
size_t rope_lastpos = -1;

const char * norm_cfg_text[] = { "NONE", "LayerNorm", "RMSNorm" };

const char * ffn_nl_type_text[] = { "", "RELU", "GELU_SIGMOID", "GELU_TANH", "SILU_LLAMA" };

const char * bias_cfg_text[] = { "OFF", "ON" };

float temperature = DEFAULT_TEMPERATURE;	//default temperature = 1.0 means "no change to the final distribution of probabilities over all the vocabulary entries"
float top_p = DEFAULT_TOP_P;			//default top_p = 0.9 means "let's sample over the top vocabulary entries whose probabilities sum up to 90%"
int top_k = -1;					//default top_k = -1 means "top_k disabled"

Tokenizer toki;

char num2str_buf[64];

bool chat_text_bold = false;
bool chat_text_italic = false;
bool chat_text_code = false;

Picture * vision_picture = NULL;

const char * jsontype_text[] = { "NULL","STRING","NUMBER","BOOL","OBJECT","ARRAY","UNKNOWN" };

long int start_ts;
long int last_ts;

struct adamw_cfg adamw_cfg;


// ============================================================
//  Unit tests, help, and batch utilities
// ============================================================

void do_unit_tests(){

	mylog(LOG_VERBOSE_DEBUG,"\n\nsizeof(float) = %d \t sizeof(size_t) = %d\n",sizeof(float),sizeof(size_t));
/*
	mylog(LOG_DEBUG,"matmulf_nn() = %d",test_matmulf_nn());
	mylog(LOG_DEBUG,"matmulf_nt() = %d",test_matmulf_nt());
*/
	return;
}

const char banner[] = "TRiP engine for Transformer Vision and Language Models, inference & training - " TRIP_VERSION_NUMBER_STRING " - by Carlo Valenti";

void print_help(){
    printf("TRiP is an all-in-one, C-based engine for transformer AI models, handling both inference and training, complete with tokenizer creation, chat functionality, and support for vision.\n\n");
    printf("USAGE:\n  ./trip <ACTION> [OPTIONS...]\n\n");

    printf("--- MAIN ACTIONS (select one) ---\n");
    printf("  --create                Create a new model from a configuration file and exit.\n");
    printf("  --train                 Train the model using the specified data.\n");
    printf("  --decode [</path/to/prompt.txt>]  Run inference on a prompt from a file or stdin.\n");
    printf("  --chat                  Start an interactive chat session with the model.\n");
    printf("  --vision [<image.jpg>]    Run multimodal inference with a given image. Image path can be provided after launch.\n");
    printf("  --build_vocab <data.txt> [--vocab_size <size>] [--tokenizer <path>]	Build and save/overwrite a new tokenizer vocabulary from a text file.\n\n");

    printf("--- MODEL & TOKENIZER OPTIONS ---\n");
    printf("  --checkpoint <path>       Path to the model checkpoint file(s) (e.g., model.safetensors).\n");
    printf("                            (default: \"model.safetensors\")\n");
    printf("  --checkpoint_type <type>  Specify the checkpoint format. Types: SAFETENSORS, LLAMA2_AK, GPT2_AK.\n");
    printf("                            (default: SAFETENSORS)\n");
    printf("  --configuration <path>    Path to the model's config.json file (used with SAFETENSORS).\n");
    printf("  --tokenizer <path>        Path to the tokenizer file.\n");
    printf("                            (default: \"tokenizer.json\")\n");
    printf("  --tokenizer_format <type> Specify the tokenizer format. Types: JSON_HUGGINGFACE, LLAMA2_AK, GPT2_AK.\n");
    printf("                            (default: JSON_HUGGINGFACE)\n");
    printf("  --tokenizer_type <type>   Specify the tokenizer algorithm. Types: SENTENCEPIECE, TRIP.\n");
    printf("                            (default: SENTENCEPIECE)\n\n");

    printf("--- INFERENCE & SAMPLING OPTIONS ---\n");
    printf("  --input_text \"<prompt>\"   Provide a text prompt directly on the command line.\n");
    printf("  --system_prompt \"<text>\"  Set a system prompt for chat mode (e.g., \"You are a helpful assistant.\").\n");
    printf("  --chat_scheme <scheme>    Specify a chat template. Schemes: LLAMA, TINY_LLAMA, GEMMA.\n");
    printf("  --chat_save_context <file> Saves the initial chat context to a file and exits.\n");
    printf("  --chat_load_context <file> Loads a chat session from a context file for faster startup.\n");
    printf("  --temperature <value>     Set the sampling temperature (e.g., 0.7). 0.0 for greedy decoding.\n");
    printf("                            (default: %.1f)\n", DEFAULT_TEMPERATURE);
    printf("  --top_p <value>           Set nucleus sampling probability (e.g., 0.9).\n");
    printf("                            (default: %.1f)\n", DEFAULT_TOP_P);
    printf("  --top_k <value>           Use top-k sampling instead of top-p. Disabled by default.\n");
    printf("  --ram                     Optimize for RAM usage by memory-mapping weights.\n\n");

    printf("--- TRAINING OPTIONS ---\n");
    printf("  --train_config <path>     Path to the training arguments JSON file.\n");
    printf("                            (default: \"training_args.json\")\n");
    printf("  --train_data <path>       Path to the training data text file.\n");
    printf("                            (default: \"training_data.txt\")\n\n");

    printf("--- MISCELLANEOUS ---\n");
    printf("  --utest                 Perform unit tests and exit.\n");
    printf("  --help                  Show this help message and exit.\n\n");

    printf("--- EXAMPLES ---\n");
    printf("  # Chat with a Gemma model\n");
    printf("  ./trip --chat --checkpoint_type SAFETENSORS --checkpoint gemma-2b-it/model.safetensors --tokenizer gemma-2b-it/tokenizer.json\n\n");
    printf("  # Run inference on a prompt\n");
    printf("  ./trip --decode --input_text \"The capital of Italy is\"\n\n");
    printf("  # Run vision (PaliGemma only) with RAM optimizations switched ON; user will be asked for input image and prompt, since none of them is specified\n");
    printf("  ./trip --ram --vision --checkpoint model-00001-of-00003.safetensors\n\n");
    printf("  # Start training\n");
    printf("  ./trip --train --checkpoint my_model/model.safetensors --tokenizer my_model/tokenizer.json --train_data my_dataset.txt\n\n");

    printf("\n\n");
}



void alloc_tokens_batch(int *** _input_tokens, int ** _n_input_tokens, size_t B, size_t T){
	
	*_input_tokens   = calloc( B , sizeof(int *) );
	*_n_input_tokens = calloc( B , sizeof(int)   );	//alloc and initialize to "zero tokens"

	for(size_t b = 0; b < B; b++){
		(*_input_tokens)[b] = calloc( T , sizeof(int));
	}
}

void free_tokens_batch(int *** _input_tokens, int ** _n_input_tokens, size_t B, size_t T){

	for(size_t b = 0; b < B; b++){
		if((*_input_tokens)[b] != NULL){
			free((*_input_tokens)[b]);
			(*_input_tokens)[b] = NULL;
		}
	}

	if(*_input_tokens != NULL){
		free(*_input_tokens);
		*_input_tokens = NULL;
	}
	if(*_n_input_tokens != NULL){
		free(*_n_input_tokens);
		*_n_input_tokens = NULL;
	}
}




void build_vocab(char * input_text_file, int size, char * input_tokenizer){

	toki.sorted_vocab = NULL;
	toki.max_token_length = MAX_TOKEN_LEN;
	toki.pad_id = -1;
	toki.bos_id = -1;
	toki.eos_id = -1;

	//let's create the data structures
	build_tokenizer(NULL, &toki, input_tokenizer, size);

	//now let's create the vocabulary
	create_vocab(&toki,input_text_file);
	dump_vocab(&toki);

	//we test the vocabulary over the same input file from which we created the vocabulary itself - neat! :D
	alloc_tokens_batch(&input_tokens, &n_input_tokens, 1, MAX_INPUT_TOKENS);

	text2tokens(NULL, &toki, input_text_file, 0x80000001, input_tokens[0], &n_input_tokens[0]);

	free_tokens_batch(&input_tokens, &n_input_tokens, 1, MAX_INPUT_TOKENS);

	return;
}




// ============================================================
//  Platform checks and signal handling
// ============================================================

void check_platform(int action){

	if((wtype == WTYPE_BF16) && (__GNUC__ < 13)){
		//if(action == ACTION_DECODE){
			mylog(LOG_INFO,"");
			mylog(LOG_INFO,"WARNING! This version of TRiP has been compiled with GCC version: %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
			mylog(LOG_INFO,"         The model you are trying to run uses BFLOAT16 as data type, which is NOT supported by such version of GCC.");
			mylog(LOG_INFO,"         The required action (%s) will output nonsense. Compile TRiP with GCC version 13 or higher!",action_text[action]);
			mylog(LOG_INFO,"");
			sleep(2);
		//}
	}


	mylog(LOG_INFO,"Number of processors: %d", omp_get_num_procs());
	mylog(LOG_INFO,"Max OpenMP threads: %d", omp_get_max_threads());


}

//signal handler function
void signal_handler(int sig){

	if(sig==SIGUSR1){	//SIGUSR1 = 10		ACTION: save checkpoint when training step is completed, then continue training

		runtime_actions	|= (1 << SIGUSR1);
		runtime_actions &= ~(1 << SIGUSR2);	//cancel SIGUSR2 if present
	}

	if(sig==SIGUSR2){	//SIGUSR2 = 12		 ACTION: save current checkpoint as soon as possible, then exit

		runtime_actions	|= (1 << SIGUSR2);
		runtime_actions &= ~(1 << SIGUSR1);	//cancel SIGUSR1 if present
	}

	return;
}




// ============================================================
//  main()
// ============================================================

int main(int argc, char ** argv){

	printf("\n");
	printf("\n");
	printf(banner);
	printf("\n");
	printf("\n");
	sleep(2);




	srand(time(NULL)); //seed the random number generator (it is used mainly when doing ACTION_CREATE)



	
	//let's parse the command line parameters

	int i = 0;

	while(i<argc){

		if(strcmp(argv[i],"--help")==0){
			print_help();
			exit(1);
		}
		if(strcmp(argv[i],"--utest")==0){
			do_unit_tests();
			exit(1);
		}
		else
		if(strcmp(argv[i],"--build_vocab")==0){

			i++;

			if(i==argc){
				mylog(LOG_ERROR,"missing input file for vocabulary building. Syntax:  --build_vocab <filename> [--vocab_size <size>]");
				exit(-1);
			}
			else{

				input_text_path = argv[i];

				if(	((i+2) > argc)
					||
					(strcmp(argv[i+1],"--vocab_size") != 0)	
				){

					mylog(LOG_INFO,"missing target size for vocabulary building. Using maximum %d. (Syntax:  --build_vocab <filename> [--vocab_size <size>])", MAX_VOCAB_ENTRIES);
					build_vocab_size = MAX_VOCAB_ENTRIES;
				}
				else{

					build_vocab_size = atoi(argv[i+2]);

					if(	(build_vocab_size > MAX_VOCAB_ENTRIES)
						||
						(build_vocab_size < (2+1))	//let's say: two symbols + eos
					){

						mylog(LOG_INFO,"invalid target size %d for vocabulary building. Using maximum %d.",build_vocab_size, MAX_VOCAB_ENTRIES);
						build_vocab_size = MAX_VOCAB_ENTRIES;
					}
				}

				

				tok_path = NULL;	//we bypass the default tokenizer path; 
							//we will start from an existing tokenizer file ONLY if the user specifies it with "--tokenizer"

				tokenizer_format = TOKFORMAT_JSON_HUGGINGFACE;
				tokenizer_type = TOKTYPE_SENTENCEPIECE;


				action = ACTION_CREATE_VOCABULARY;

			}
		}
		else
		if(strcmp(argv[i],"--ram")==0){
			ramflag = RAM_BASIC_OPTIMIZATIONS;
		}
		else	
		if(strcmp(argv[i],"--create")==0){

			action = ACTION_CREATE;
		}
		else	
		if(strcmp(argv[i],"--train")==0){

			action = ACTION_TRAIN;
		}
		else
		if(strcmp(argv[i],"--train_config")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing training configuration file for train_config. Syntax:  --train_config <filename>");
				exit(-1);
			}
			else{
				train_cfg_path = argv[i];
				mylog(LOG_INFO,"Using training configuration from file \"%s\".",train_cfg_path);
			}
		}
		else
		if(strcmp(argv[i],"--train_data")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing training data (text) input file for train_data. Syntax:  --train_data <filename>");
				exit(-1);
			}
			else{
				train_data_path = argv[i];
				mylog(LOG_INFO,"Using training data (text) from file \"%s\".",train_data_path);
			}
		}
		else	
		if(strcmp(argv[i],"--vision")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
			
			/*
				i--;
				if(fgets(inbuf,MAX_TEXT_LEN,stdin) == NULL)	inbuf[0] = '\0';
				int len;
				len = strlen(inbuf);
				while(inbuf[len-1]<=' ')	//removing any annoying trailing character
				{
					inbuf[len-1]='\0';
					len = strlen(inbuf);
				}
			*/

				mylog(LOG_INFO,"image file not specified. Possible usage: --vision [<image_file.jpg> [--input_text=\"(text)\"]]");
				//exit(1);
			}
			else{

				strcpy(vision_picture_path, argv[i]);


				i++;
			}




			if((i < argc) && (memcmp(argv[i],"--input_text=",13)==0)){
				int len;
				len = strlen(&argv[i][13]);
				memcpy(inbuf,&argv[i][13],len);
				if(inbuf[len-1]=='\"'){
					inbuf[len-1] = '\0';
					len--;	//if required
				}

				mylog(LOG_DEBUG,"\nText to start decoding with: \n%s",inbuf);
			}
			else{

				mylog(LOG_INFO,"No input text specified to start vision model with.");
				//exit(-1);
				inbuf[0] = '\0';	//will trigger user prompt

				i--;
			}




			action = ACTION_VISION;


		}
		else	
		if(strcmp(argv[i],"--decode")==0){

			i++;

			if((i< argc) && (memcmp(argv[i],"--input_text=",13)==0)){
				int len;
				len = strlen(&argv[i][13]);
				memcpy(inbuf,&argv[i][13],len);
				if(inbuf[len-1]=='\"'){
					inbuf[len-1] = '\0';
					len--;	//if required
				}
			}
			else
			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				i--;
				if(fgets(inbuf,MAX_TEXT_LEN,stdin) == NULL)	inbuf[0] = '\0';
				int len;
				len = strlen(inbuf);
				while(inbuf[len-1]<=' ')	//removing any annoying trailing character
				{
					inbuf[len-1]='\0';
					len = strlen(inbuf);
				}
			}
			else{
				FILE * f;

				if((f = fopen(argv[i],"r")) != NULL){
					size_t nread = fread(inbuf,sizeof(char),MAX_TEXT_LEN,f);
					inbuf[nread] = '\0';
					fclose(f);
				}
				else{
					mylog(LOG_ERROR,"can't find input file \"%s\" to start decoding with. Syntax:  --decode [<filename>]",argv[i]);
					exit(-1);
				}
			}

			mylog(LOG_DEBUG,"\nText to start decoding with: \n%s",inbuf);



			action = ACTION_DECODE;


		}
		else	
		if(strcmp(argv[i],"--chat")==0){

			i++;
			inbuf[0] = '\0';

			if(	(i< argc)
				&&
				(memcmp(argv[i],"--system_prompt=",16)==0)
				&&
				(strlen(argv[i])>17)
			){

				int len;
				len = strlen(&argv[i][17]);
				memcpy(inbuf,&argv[i][17],len);
				if((inbuf[len-1]=='\"') || (inbuf[len-1]=='\'')){
					inbuf[len-1] = '\0';
					len--;	//if required
				}
			}

			mylog(LOG_DEBUG,"\nSystem prompt: \n%s",inbuf);



			action = ACTION_CHAT;


		}
		else
		if(strcmp(argv[i],"--chat_scheme")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing value for chat_scheme. Syntax:  --chat_scheme <LLAMA|TINY_LLAMA>");
				exit(-1);
			}
			else{
				if(strcmp(argv[i],"LLAMA")==0)		chat_scheme = CHATSCHEME_LLAMA;
				else
				if(strcmp(argv[i],"TINY_LLAMA")==0)	chat_scheme = CHATSCHEME_TINYLLAMA;
				else
				if(strcmp(argv[i],"GEMMA")==0)		chat_scheme = CHATSCHEME_GEMMA;

				mylog(LOG_INFO,"Chat scheme = %s ",chatscheme_text[chat_scheme]);
			}
		}
		else
		if(strcmp(argv[i],"--chat_save_context")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing destination context file for chat_save_context. Syntax:  --chat_save_context <filename>");
				exit(-1);
			}
			else{
				chat_context_file = argv[i];
				chat_flags |= CHAT_SAVE_CONTEXT;
				mylog(LOG_INFO,"Chat context will be saved to file \"%s\".",chat_context_file);
			}
		}
		else
		if(strcmp(argv[i],"--chat_load_context")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing source context file for chat_load_context. Syntax:  --chat_load_context <filename>");
				exit(-1);
			}
			else{
				chat_context_file = argv[i];
				chat_flags |= CHAT_LOAD_CONTEXT;
				mylog(LOG_INFO,"Chat context will be sourced from file \"%s\".",chat_context_file);
			}
		}
		else
		if(strcmp(argv[i],"--top_k")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing value for top_k. Syntax:  --top_k <1 ... (vocab_size)>");
				exit(-1);
			}
			else{
				sscanf(argv[i],"%d",&top_k);
				mylog(LOG_INFO,"Top_k = %d ",top_k);
			}
		}
		else
		if(strcmp(argv[i],"--top_p")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing value for top_p. Syntax:  --top_p <0.000 ... 1.000>");
				exit(-1);
			}
			else{
				sscanf(argv[i],"%f",&top_p);
				mylog(LOG_INFO,"Top_p = %.3f ",top_p);
			}
		}
		else
		if(strcmp(argv[i],"--temperature")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR, "missing value for temperature. Syntax:  --temperature <0.000 ... (inf)>");
				exit(-1);
			}
			else{
				sscanf(argv[i],"%f",&temperature);
				mylog(LOG_INFO,"Temperature = %.3f ",temperature);
			}
		}
		else
		if(strcmp(argv[i],"--tokenizer")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing input file for tokenizer. Syntax:  --tokenizer <filename>");
				exit(-1);
			}
			else{
				tok_path = argv[i];
				mylog(LOG_INFO,"Tokenizer = %s",tok_path);
			}
		}
		else
		if(strcmp(argv[i],"--tokenizer_format")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing tokenizer_format. Syntax:  --tokenizer_format < LLAMA2_AK | JSON_HUGGINGFACE | GPT2_AK>");
				exit(-1);
			}
			else{
				if(strcmp(argv[i],"TRIP")==0)			tokenizer_format = TOKFORMAT_TRIP;
				if(strcmp(argv[i],"LLAMA2_AK")==0)		tokenizer_format = TOKFORMAT_LLAMA2_AK;
				if(strcmp(argv[i],"JSON_HUGGINGFACE")==0)	tokenizer_format = TOKFORMAT_JSON_HUGGINGFACE;
				if(strcmp(argv[i],"GPT2_AK")==0)		tokenizer_format = TOKFORMAT_GPT2_AK;


				mylog(LOG_INFO,"Tokenizer_format = %s",tokformat_text[tokenizer_format]);
			}
		}
		else
		if(strcmp(argv[i],"--tokenizer_type")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing tokenizer_type. Syntax:  --tokenizer_format < TRIP | SENTENCEPIECE >");
				exit(-1);
			}
			else{
				if(strcmp(argv[i],"TRIP")==0)			tokenizer_type = TOKTYPE_TRIP;
				if(strcmp(argv[i],"SENTENCEPIECE")==0)		tokenizer_type = TOKTYPE_SENTENCEPIECE;


				mylog(LOG_INFO,"Tokenizer_type = %s",toktype_text[tokenizer_type]);
			}
		}
		else
		if(strcmp(argv[i],"--configuration")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing configuration file. Syntax:  --configuration <filename>");
				exit(-1);
			}
			else{
				cfg_path = argv[i];
				mylog(LOG_INFO,"Configuration = %s",cfg_path);
			}
		}
		else
		if(strcmp(argv[i],"--checkpoint")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing checkpoint file. Syntax:  --checkpoint <filename>");
				exit(-1);
			}
			else{
				mod_path = argv[i];
				mylog(LOG_INFO,"Model = %s",mod_path);
			}
		}
		else
		if(strcmp(argv[i],"--checkpoint_type")==0){

			i++;

			if((i==argc) || (memcmp(argv[i],"--",2)==0)){
				mylog(LOG_ERROR,"missing checkpoint_type. Syntax:  --checkpoint_type < TRIP | LLAMA2_AK | SAFETENSORS | GPT2_AK >");
				exit(-1);
			}
			else{
				if(strcmp(argv[i],"LLAMA2_AK")==0)	checkpoint_type = CP_LLAMA2_AK;
				if(strcmp(argv[i],"SAFETENSORS")==0)	checkpoint_type = CP_SAFETENSORS;
				if(strcmp(argv[i],"GPT2_AK")==0)	checkpoint_type = CP_GPT2_AK;


				mylog(LOG_INFO,"Checkpoint_type = %s",cp_litteral[checkpoint_type]);
			}
		}


		i++;
	}

	//end of command parser




	//let's register one signal handler for all the signals to be handled

	//during training: will save the current checkpoint as soon as possible, and then training will continue
	struct sigaction sa;		//SIGUSR1, code 10
	sa.sa_handler = signal_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	//list of the signals handled by the single signal handler signal_handler
	sigaction(SIGUSR1, &sa, NULL);		//10
	sigaction(SIGUSR2, &sa, NULL);		//12





	//now, let's perform the action requested by the user!

	if(action == ACTION_CREATE_VOCABULARY){


		build_vocab(input_text_path, build_vocab_size, tok_path);

		char * outfile = calloc((strlen(input_text_path) + strlen(".tokenizer") + 1), sizeof(char));
		sprintf(outfile,"%s%s",input_text_path,".tokenizer");
		save_tokenizer(&toki,outfile,toki.vocab_size);

		free(outfile);
		exit(1);

	}
	else
	if(action == ACTION_CREATE){

		checkpoint_type = CP_SAFETENSORS;


		Model * trip = (Model *)myalloc(sizeof(Model));

		//let's initialize the model from the configuration file
		trip->config.submodel_type = MODELTYPE_DECODER;
		init_model(trip, checkpoint_type, NULL);
		mylog(LOG_INFO,"Model initialized.");


		mylog(LOG_INFO,"Saving new model...");

		save_model(trip);

		
		mylog(LOG_INFO,"NEW MODEL SAVED! Exiting...");
		exit(1);
	}
	else
	if(action == ACTION_TRAIN){

		
		calculate_loss = true;


		Model * trip = (Model *)myalloc(sizeof(Model));

		
		//let's load the checkpoint for the model
		load_model_from_checkpoint(trip, mod_path);
		mylog(LOG_INFO,"Model loaded from checkpoint.");

		

		//let's initialize the model from the checkpoint we have just loaded;
		//this means:
		//
		trip->config.submodel_type = MODELTYPE_DECODER;
		init_model(trip, checkpoint_type, NULL);
		mylog(LOG_INFO,"Model initialized.");



		////print the configuration of the sampler
		//print_sampler_configuration();



		//let's load the tokenizer parameters, and build it
		build_tokenizer(trip, &toki, tok_path, trip->config.vocab_size);
		mylog(LOG_INFO,"Tokenizer built.");



		//let's do a quick check on the platform we are running on (GCC, CPU...)
		check_platform(action);



		//let's initialize the AdamW optimizer
		adamw_init();

		//let's load the TRAINING CONFIGURATION! This will configure the AdamW optimizer + training schedule details + more
		load_training_args(train_cfg_path);



	FILE * training_data = NULL;

	if( (training_data = fopen(train_data_path, "rb"))  ==  NULL){
		mylog(LOG_ERROR, "Failed to open training data file \"%s\". Exiting...", train_data_path);
                exit(1);		
	}
	else{
		mylog(LOG_INFO, "Training data file \"%s\" opened successfully.", train_data_path);
		fseek(training_data, 0, SEEK_SET);	
	}



	bool tobe_saved = false;

	float training_max_lr = adamw_cfg.learning_rate; //let's save the max learning rate


	ssize_t step = 0;
	ssize_t epoch = 0;

	


	//THE TRAINING LOOP!
	while(	((step < training_max_steps)  ||  (training_max_steps <= 0))
		&&
		((epoch < training_num_epochs)  ||  (training_num_epochs <= 0))
	){


		mylog(LOG_INFO,"training step: %zu / %zu", step, training_max_steps);


		if(strcmp(training_lr_scheduler_type, "cosine") == 0){

			adamw_cfg.learning_rate = cosine_annealing_lr(
		            step,
		            training_warmup_steps,
		            training_max_steps,
		            training_max_lr,
		            training_min_lr
		        );

			


		        mylog(LOG_INFO, "Step %zu: LR set to %f", step, adamw_cfg.learning_rate);


		}


		size_t batch_size = training_batch_size;	//batch_size will be the one we just read the from the training configuration file
		size_t max_tokens = (size_t)(trip->config.sequence_maxtokens);


		alloc_tokens_batch(&input_tokens, &n_input_tokens, batch_size, max_tokens);


		char * tempbuf = myalloc(MAX_TEXT_LEN * sizeof(char));

		size_t b = 0;

		while(b < batch_size){

			char * p;

			while(	(b < batch_size)
				&&
				((p=fgets(tempbuf,MAX_TEXT_LEN,training_data)) != NULL)
			){

    				tempbuf[MAX_TEXT_LEN-1] = '\0';

				if(strlen(tempbuf) <= 1){

					mylog(LOG_VERBOSE_DEBUG,"(skipping empty line in training data...)");

					continue;
				}
				else{
					mylog(LOG_INFO,"Text sequence %d/%d: \"%s\"",b,batch_size,tempbuf);
				}

				tempbuf[strcspn(tempbuf, "\n")] = '\0';	//let's strip the newline character at the end of the line (if present)


				text2tokens(trip, &toki, tempbuf, 0x00000001, input_tokens[b], &n_input_tokens[b]); 
				mylog(LOG_DEBUG,"Text sequence %d/%d tokenized.",b,batch_size);


				b++;				
			}


			if(b < batch_size){	//if we just completed one full reading of the input file, let's restart it (new epoch)

				fseek(training_data, 0, SEEK_SET);
				
				epoch ++ ;
			}
			
		}
		
		free(tempbuf);





		size_t max_n_input_tokens;
		//let's search for the maximum number of tokens in sequences
		max_n_input_tokens = 0;
		for(int i = 0; i < batch_size; i++){
			if(n_input_tokens[i] > max_n_input_tokens){
				max_n_input_tokens = n_input_tokens[i];
			}
		}

		//let's explicitly put the pad token in every extra token beyond the length of each sequence - as an extra safety measure 
		for(size_t b = 0; b < batch_size; b++){
			for(size_t t = n_input_tokens[b]; t < max_n_input_tokens; t++){
				input_tokens[b][t] = toki.pad_id;
			}
			////let's update the length to be uniform for all sequences in the batch
			//n_input_tokens[b] = max_n_input_tokens;
		}	

		//NOTE:
		//we set the sequence size to  "max tokens" - 1  because, when training, the last token in the input sequence has not to be processed;
		//it serves only as the target against which we will calculate loss, 
		//given the predicted probability calculated for that next "target" token after the processing of the current token.

		size_t sequence_size = (max_n_input_tokens - 1);	



		alloc_forward_memory(trip, batch_size, sequence_size, ACTION_TRAIN);
		alloc_kv_memory(trip, batch_size, sequence_size, ACTION_TRAIN);


		//let's decode! We'll stop when the decoder generates an EOS token (outside the input sequence)
		forward(trip, NULL, NULL, input_tokens, n_input_tokens, 0, ATTENTION_CAUSAL, batch_size, sequence_size);


		alloc_gradients_memory(trip, batch_size, sequence_size);

		backward(trip, ATTENTION_CAUSAL, input_tokens, batch_size, sequence_size);


		gradients_check(trip, sequence_size);


		bool updated;
		updated = model_update(trip, step, sequence_size);
		tobe_saved = (tobe_saved || updated);

		free_gradients_memory(trip, batch_size, sequence_size);



		//let's check if it is time to save a checkpoint
	
		if(runtime_actions & (1<<SIGUSR1)){	//10
			save_model(trip);
			runtime_actions &= ~(1<<SIGUSR1);
			tobe_saved = false;
			//break;	//continue training!
		}
		else
		if(runtime_actions & (1<<SIGUSR2)){	//12
			if(tobe_saved){
				save_model(trip);
				tobe_saved = false;
			}
			break;		//exit!
		}
		else
		if( (training_save_steps  >  0)
		    &&
		    (((step + 1) % training_save_steps)  ==  0)
		){
	       	        if(tobe_saved){
                	    mylog(LOG_INFO, "Saving model at step %zu...", step);
	                    save_model(trip);
        	            tobe_saved = false;
	                }
	        }



		//let's free up the dynamically-allocated memory
		free_forward_memory(trip);
		free_kv_memory(trip);

		free_tokens_batch(&input_tokens, &n_input_tokens, batch_size, MAX_INPUT_TOKENS);



		step ++ ;	
	}


	fclose(training_data);





		if(tobe_saved){
			save_model(trip);
			tobe_saved = false;
		}



		//forward(trip, NULL, NULL, input_tokens, n_input_tokens, 0, ATTENTION_CAUSAL, batch_size, sequence_size);


	



		adamw_free();



/*

	//TEST
	//let's chat!
	action = ACTION_CHAT;
	log_cfg = LOG_INFO;
	calculate_loss = false; 

	alloc_kv_memory(trip, 1, max_tokens, ACTION_CHAT);


	strcpy(inbuf,"You are a helpful assistant.");
	//strcpy(inbuf," ");


	chat(trip,inbuf);
*/




		free_model_memory(trip);

		exit(1);

	}
	else
	if(action == ACTION_VISION){

	
		Model * trip_encoder = (Model *)myalloc(sizeof(Model));
		Model * trip_decoder = (Model *)myalloc(sizeof(Model));


		//let's load the checkpoint for the model's vision encoder
		load_model_from_checkpoint(trip_encoder, mod_path);
		mylog(LOG_INFO,"Model vision encoder loaded from checkpoint.");


		//let's initialize the model from the checkpoint we have just loaded;
		//this means:
		//
		trip_encoder->config.submodel_type = MODELTYPE_VISION_ENCODER;
		init_model(trip_encoder, checkpoint_type, NULL);
		mylog(LOG_INFO,"Model vision encoder initialized.");




		//let's load the checkpoint for the model's language decoder
		load_model_from_checkpoint(trip_decoder, mod_path);
		mylog(LOG_INFO,"Model language decoder loaded from checkpoint.");


		//let's initialize the model from the checkpoint we have just loaded;
		//this means:
		//
		trip_decoder->config.submodel_type = MODELTYPE_DECODER;
		init_model(trip_decoder, checkpoint_type, trip_encoder);	//we specify here (last argument: "trip_encoder")
										//that the decoder has a brother model, which is the encoder, 
										//so that they can share the same weights RAM area / files
		mylog(LOG_INFO,"Model language decoder initialized.");



		//print the configuration of the sampler
		print_sampler_configuration();


		//let's do a quick check on the platform we are running on (GCC, CPU...)
		check_platform(action);
		


		//let's load the tokenizer parameters, and build it
		build_tokenizer(trip_decoder, &toki, tok_path, trip_decoder->config.vocab_size);
		mylog(LOG_INFO,"Tokenizer built.");


/***************

PALIGEMMA ARCHITECTURE
 
Here, we must:
- load the jpg file
- adjust it to the encoder input format
- give the picture as input to the encoder
- collect the soft tokens output by the encoder, 
  and put them as the initial part of the sequence which will be given as input to the decoder

**************/

	byte * encoder_input_streams;
	byte * encoder_output_streams;
	byte * decoder_input_streams;

	size_t max_tokens = (size_t)(trip_decoder->config.sequence_maxtokens);
	size_t n_patches  = (size_t)(trip_encoder->config.vision_image_tokens);

	alloc_tokens_batch(&input_tokens, &n_input_tokens, 1, max_tokens);

	alloc_kv_memory(trip_encoder, 1,  n_patches, ACTION_VISION);
	alloc_kv_memory(trip_decoder, 1, max_tokens, ACTION_VISION);



	while(1){

		while(vision_picture == NULL){

			while(vision_picture_path[0]=='\0'){
				user_prompt[0] = '\0';				
				printf("\r\nPlease specify the path of the picture:  ");
				get_userprompt( endstr(user_prompt), (PROMPT_MAXLEN-strlen(user_prompt)-10) );
				if(user_prompt[strlen(user_prompt)-1] == '\n')	user_prompt[strlen(user_prompt)-1] = '\0';
				strcpy(vision_picture_path, user_prompt);		
			}

			
			vision_picture = (Picture *)malloc(sizeof(Picture));
			vision_picture->pic = (Pixel *)read_jpeg_to_rgb(vision_picture_path, &vision_picture->width, &vision_picture->height);

			if(vision_picture->pic == NULL){

				printf("\r\nCan't read image file \"%s\" to start vision model with.\r\n",vision_picture_path);
#ifdef TRIP_DEBUG
				mylog(LOG_INFO,"can't read image file \"%s\" to start vision model with.",vision_picture_path);
#endif

				vision_picture_path[0] = '\0';
				free(vision_picture);
				vision_picture = NULL;
			}
			else{

				printf("\r\nImage file \"%s\" loaded.\r\n",vision_picture_path);
#ifdef TRIP_DEBUG
				mylog(LOG_INFO,"Image file \"%s\" loaded.",vision_picture_path);
#endif
			}

		}

		//this is just to display the chosen picture, resized to a fixed standard 600x600 size;
		//the picture is not being resized in memory here;
		//it will be resized in memory to the model input picture size in picture2streams();

		#define	PIC_NICESIZE	600

		displayPicture_resize(vision_picture,PIC_NICESIZE,PIC_NICESIZE, -1);


		//let's resize the picture to the model input picture size, and translate it to an input stream for the ENCODER
		encoder_input_streams = picture2streams(trip_encoder, vision_picture);


		size_t n_patches  = trip_encoder->config.vision_image_tokens;
		size_t encoder_dim_stream = trip_encoder->config.dim_stream;
		size_t decoder_dim_stream = trip_decoder->config.dim_stream;


#ifdef TRIP_DEBUG
wdebug(encoder_input_streams,WTYPE_FLOAT32,12,"first patch first 12 values after positional embeddings",-1,0);
wdebug(&encoder_input_streams[1*encoder_dim_stream*sizeof(float)],WTYPE_FLOAT32,12,"second patch first 12 values after positional embeddings",-1,1);
#endif


		//let's ENCODE!
		encoder_output_streams = (byte *)myalloc(1 * n_patches * encoder_dim_stream * sizeof(float));
		//encoder will populate only the first n_patches streams
		forward(trip_encoder, encoder_input_streams, encoder_output_streams, NULL, NULL, 0, ATTENTION_FULL, 1, n_patches);



		while(1){
			//now, let's tokenize the input text (will be appended to the soft tokens generated by the encoder);
			//we do it now to know how much memory we need to alloc in total
			n_input_tokens[0] = n_patches;	//we already have the length of the soft tokens from the encoder at the beginning of the sequence


			while(inbuf[0]=='\0'){
				user_prompt[0] = '\0';				
				printf("\r\nPROMPT:  ");
				get_userprompt( endstr(user_prompt), (PROMPT_MAXLEN-strlen(user_prompt)-10) );		
				if(user_prompt[strlen(user_prompt)-1] == '\n')	user_prompt[strlen(user_prompt)-1] = '\0';		
				strcpy(inbuf,user_prompt);			
			}


			if(	(strcmp(inbuf,"pic")==0)
				||
				(strcmp(inbuf,"newpic")==0)
				||
				(strcmp(inbuf,"new")==0)
			){
				vision_picture_path[0] = '\0';
				inbuf[0] = '\0';
				if(vision_picture->pic != NULL){free(vision_picture->pic);}
				if(vision_picture      != NULL){free(vision_picture     );}
				vision_picture = NULL;
				break;
			}
			else{
	
				text2tokens(trip_decoder, &toki, inbuf, 0x00000009, input_tokens[0], &n_input_tokens[0]);	//0x1: add BOS; +0x8: add "\n" at the end, if not present
				mylog(LOG_DEBUG,"Text tokenized.");
	
	
				byte * decoder_input_streams = (byte *)malloc(n_input_tokens[0] * decoder_dim_stream * sizeof(float));
	
		
				//let's do the (LINEAR) projection of the output of the encoder to the input of the decoder
				matmulf_nt(	&trip_encoder->w.multimodal_projector_w[0],
						&encoder_output_streams[0], 
						encoder_dim_stream,decoder_dim_stream, 
						encoder_dim_stream,n_patches, &decoder_input_streams[0]
				);
		
		
				for(size_t ppos = 0; ppos < n_patches; ppos++){
					sum_vectors(	&decoder_input_streams[ppos*decoder_dim_stream*sizeof(float)], WTYPE_FLOAT32, 
							&trip_encoder->w.multimodal_projector_b[0], wtype, 
							decoder_dim_stream, 
							&decoder_input_streams[ppos*decoder_dim_stream*sizeof(float)]
					);
				}
		
				float scale_factor = 1.0 / sqrtf((float)decoder_dim_stream);			
		
				multiply_vector(&decoder_input_streams[0], (decoder_dim_stream*n_patches), scale_factor, &decoder_input_streams[0]);			
		
		
		
#ifdef TRIP_DEBUG
wdebug(decoder_input_streams,WTYPE_FLOAT32,12,"first patch first 12 values after multimodal projector",-1,1);
#endif
		
		
			
				//let's concatenate to the projected output of the encoder the embeddings translation of the text tokens
			    for(size_t ppos = n_patches; ppos < n_input_tokens[0]; ppos++){
		
		
		
				if(wtype == WTYPE_FLOAT32){
		
#ifdef TRIP_DEBUG
mylog(LOG_INFO,"Copying embeddings for token[%d]=%d",ppos,input_tokens[ppos]);
#endif
		
					memcpy(&decoder_input_streams[ppos*decoder_dim_stream*wsize], &trip_decoder->w.embeddings[((size_t)input_tokens[0][ppos]) * decoder_dim_stream * wsize], (decoder_dim_stream * wsize));
				}
				else
				if(wtype == WTYPE_BF16){
					for(size_t i = 0; i < decoder_dim_stream; i++){
						__bf16 emb_i = ((__bf16 *)&trip_decoder->w.embeddings[((size_t)input_tokens[0][ppos]) * decoder_dim_stream * wsize])[i];
						((float *)&decoder_input_streams[ppos*decoder_dim_stream*sizeof(float)])[i] = (float)emb_i;
					}
				}
				else
				if(wtype == WTYPE_FLOAT16){
					for(size_t i = 0; i < decoder_dim_stream; i++){
						_Float16 emb_i = ((_Float16 *)&trip_decoder->w.embeddings[((size_t)input_tokens[0][ppos]) * decoder_dim_stream * wsize])[i];
						((float *)&decoder_input_streams[ppos*decoder_dim_stream*sizeof(float)])[i] = (float)emb_i;
					}
				}
			    }
		
		
		
				
		
				//let's DECODE:
				// -  we start with FULL ATTENTION over the encoder output embeddings + the text embeddings;
				// -  then, we will switch automatically to CAUSAL ATTENTION (the switching is managed within function forward() automatically)

				size_t pos = 0;

				while(input_tokens[0][pos] != toki.eos_id){ 

					//in PaliGemma, at the beginning we must process image soft tokens + text prefix altogether with full attention, 
					//and then we go on with causal attention
					int attention_type = ((pos==0) ? ATTENTION_FULL : ATTENTION_CAUSAL);
					int parallel_steps = ((pos==0) ? n_input_tokens[0] : 1);

					pos = forward(trip_decoder, decoder_input_streams, NULL, input_tokens, n_input_tokens, pos, attention_type, 1, (size_t)parallel_steps);

					if(vision_detect_status == 4){

						Picture * detect_picture = (Picture *)malloc(sizeof(Picture));
			                        detect_picture->pic = (Pixel *)read_jpeg_to_rgb(vision_picture_path, &detect_picture->width, &detect_picture->height);

						Pixel * tempix;
						tempix = (Pixel *)resize_rgb_lanczos((unsigned char *)detect_picture->pic, detect_picture->width, detect_picture->height, PIC_NICESIZE, PIC_NICESIZE);
						free(detect_picture->pic);
						detect_picture->pic = tempix;
						detect_picture->width = PIC_NICESIZE;
						detect_picture->height = PIC_NICESIZE;

						vision_detect_y_min = (int)((((float)vision_detect_y_min)/1024.0) * (float)detect_picture->height);
						vision_detect_x_min = (int)((((float)vision_detect_x_min)/1024.0) * (float)detect_picture->width);
						vision_detect_y_max = (int)((((float)vision_detect_y_max)/1024.0) * (float)detect_picture->height);
						vision_detect_x_max = (int)((((float)vision_detect_x_max)/1024.0) * (float)detect_picture->width);

						draw_rectangle(detect_picture, vision_detect_y_min,vision_detect_x_min,vision_detect_y_max,vision_detect_x_max, 0,255,0, 2,0.2);

						displayPicture(detect_picture, -1);

						free(detect_picture->pic);
						free(detect_picture     );

						vision_detect_y_min = 0;
						vision_detect_x_min = 0;
						vision_detect_y_max = 0;
						vision_detect_x_max = 0;

						vision_detect_status = 0;
					}
				}

				


				//let's invalidate the last prompt; we will be requested for a new one
				inbuf[0] = '\0';
			}	
		}	
	}


#ifdef TRIP_DEBUG
mylog(LOG_INFO,"EARLY EXIT!");
#endif
//exit(1);

		//let's free up the dynamically-allocated memory
		if(encoder_input_streams  != NULL)	free(encoder_input_streams);
		if(encoder_output_streams != NULL)	free(encoder_output_streams);
		if(decoder_input_streams  != NULL)	free(decoder_input_streams);

		free_kv_memory(trip_encoder);
		free_kv_memory(trip_decoder);
		free_model_memory(trip_encoder);
		free_model_memory(trip_decoder);

		free_tokens_batch(&input_tokens, &n_input_tokens, 1, MAX_INPUT_TOKENS);

		printf("\n\n");
		exit(1);
	}
	else
	if(action == ACTION_DECODE){

		Model * trip = (Model *)myalloc(sizeof(Model));
		
		//let's load the checkpoint for the model
		load_model_from_checkpoint(trip, mod_path);
		mylog(LOG_INFO,"Model loaded from checkpoint.");


		//let's initialize the model from the checkpoint we have just loaded;
		//this means:
		//
		trip->config.submodel_type = MODELTYPE_DECODER;
		init_model(trip, checkpoint_type, NULL);
		mylog(LOG_INFO,"Model initialized.");



		//print the configuration of the sampler
		print_sampler_configuration();


		//let's load the tokenizer parameters, and build it
		build_tokenizer(trip, &toki, tok_path, trip->config.vocab_size);
		mylog(LOG_INFO,"Tokenizer built.");


		
		//let's tokenize the input text
		size_t max_tokens = (size_t)(trip->config.sequence_maxtokens);
		alloc_tokens_batch(&input_tokens, &n_input_tokens, 1, max_tokens);
		text2tokens(trip, &toki, inbuf, 0x00000001, input_tokens[0], &n_input_tokens[0]); 
		mylog(LOG_DEBUG,"Text tokenized.");

		alloc_kv_memory(trip, 1, max_tokens, ACTION_DECODE);


		//let's perform a quick check on the platform we are running on (GCC, CPU...)
		check_platform(action);


		//let's decode! We'll stop when the decoder generates an EOS token (out of the input sequence)
		forward(trip, NULL, NULL, input_tokens, n_input_tokens, 0, ATTENTION_CAUSAL, 1, (size_t)n_input_tokens[0]);


		//let's free up the dynamically-allocated memory
		free_kv_memory(trip);
		free_model_memory(trip);
		free_tokens_batch(&input_tokens, &n_input_tokens, 1, MAX_INPUT_TOKENS);

		printf("\n\n");
		exit(1);
	}
	else
	if(action == ACTION_CHAT){

		log_cfg = LOG_INFO;

		Model * trip = (Model *)myalloc(sizeof(Model));

		//let's load the checkpoint for the model
		load_model_from_checkpoint(trip, mod_path);
		mylog(LOG_INFO,"Model loaded from checkpoint.");


		//let's initialize the model from the checkpoint we have just loaded;
		//this means:
		//
		trip->config.submodel_type = MODELTYPE_DECODER;
		init_model(trip, checkpoint_type, NULL);
		mylog(LOG_INFO,"Model initialized.");



		//print the configuration of the sampler
		print_sampler_configuration();


		//let's load the tokenizer parameters, and build it
		build_tokenizer(trip, &toki, tok_path, trip->config.vocab_size);
		mylog(LOG_INFO,"Tokenizer built.");


		size_t max_tokens = (size_t)(trip->config.sequence_maxtokens);
		alloc_kv_memory(trip, 1, max_tokens, ACTION_CHAT);


		//let's do a quick check on the platform we are running on (GCC, CPU...)
		check_platform(action);


		
		//let's chat!		
		chat(trip,inbuf);



		//let's free up the dynamically-allocated memory
		free_kv_memory(trip);
		free_model_memory(trip);

		printf("\n\n");
		exit(1);
	}
	
}




// ============================================================
//  Chat system
// ============================================================

void chat_save_context(Model * model, int * intok, int * n_intoks){

			size_t max_tokens = (size_t)(model->config.sequence_maxtokens);
			size_t vocab_size = (size_t)(model->config.vocab_size);
			size_t n_layers   = (size_t)(model->config.n_layers);
			size_t n_keys     = (size_t)(model->config.n_keys);
			size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);


			FILE * f;

			f = fopen(chat_context_file, "wb");

			if(f == NULL){
				mylog(LOG_ERROR, "Failed to open chat context file \"%s\" for save. Exiting...", chat_context_file);
				exit(1);
			}

			fprintf(f,"%d:",*n_intoks);	//first, let's save how many tokens is the context

			for(size_t i = 0; i < *n_intoks; i++){	//then, let's save the list of the token IDs (not the text, to avoid different tokenization issues)
				fprintf(f,"%d,",intok[i]);
			}

			//fprintf(f,"%c",0x00);	//string terminator
			

			for(size_t layer = 0; layer < n_layers; layer++){					
				for(size_t key = 0; key < n_keys; key++){					
					fwrite(&model->fm.keys[layer][key*max_tokens*dim_qkv*sizeof(float)],1,((*n_intoks)*dim_qkv*sizeof(float)),f);   //let's save the key   cache
					fwrite(&model->fm.values[layer][key*max_tokens*dim_qkv*sizeof(float)],1,((*n_intoks)*dim_qkv*sizeof(float)),f); //let's save the value cache
				}
			}


			fclose(f);

			return;
}


void chat_load_context(Model * model, int * intok, int * n_intoks){

			size_t max_tokens = (size_t)(model->config.sequence_maxtokens);
			size_t vocab_size = (size_t)(model->config.vocab_size);
			size_t n_layers   = (size_t)(model->config.n_layers);
			size_t n_keys     = (size_t)(model->config.n_keys);
			size_t dim_qkv 	  = (size_t)(model->config.dim_stream / model->config.n_queries);


			FILE * f;

			f = fopen(chat_context_file, "rb");

			if(f == NULL){
				mylog(LOG_ERROR, "Failed to open chat context file \"%s\" for read. Exiting...", chat_context_file);
				exit(1);
			}


			if(fscanf(f,"%d:",n_intoks) != 1){	//first, let's read how many tokens were saved as the context
				mylog(LOG_ERROR,"Error reading number of context tokens from file \"%s\". Exiting...",chat_context_file);
				fclose(f);
				exit(1);
			}




			for(size_t i = 0; i < *n_intoks; i++){	//then, let's save the list of the token IDs (not the text, to avoid different tokenization issues)
				if(fscanf(f,"%d,",&intok[i]) != 1){
					mylog(LOG_ERROR,"Error reading context token ID #%d from file \"%s\". Exiting...",i,chat_context_file);
					fclose(f);
					exit(1);
				}
			}
			
			//fgets(inbuf,MAX_TEXT_LEN,f);	//to consume the string terminator


#ifdef TRIP_DEBUG
mylog(LOG_INFO,"Remaining header: %s",inbuf);
#endif
			
			size_t kv_layer_contextsize;
			kv_layer_contextsize = ((*n_intoks)*dim_qkv*sizeof(float));

			for(size_t layer = 0; layer < n_layers; layer++){
				for(size_t key = 0; key < n_keys; key++){

					//let's read the key cache for this layer and key/value head
					if( fread(&model->fm.keys[layer][key*max_tokens*dim_qkv*sizeof(float)], 1, kv_layer_contextsize, f)  !=  kv_layer_contextsize ){
						mylog(LOG_ERROR,"bad size while reading key cache  @  layer %d  head %d. Exiting...",layer,key);
						exit(1);
					}
					//let's read the value cache for this layer and key/value head
					if( fread(&model->fm.values[layer][key*max_tokens*dim_qkv*sizeof(float)], 1, kv_layer_contextsize, f)  !=  kv_layer_contextsize ){
						mylog(LOG_ERROR,"bad size while reading value cache  @  layer %d  head %d. Exiting...",layer,key);
						exit(1);
					}
				}
			}


			fclose(f);

			mylog(LOG_INFO,"Chat context file \"%s\" successfully loaded: %d tokens retrieved, key cache and value cache OK", chat_context_file);

			return;
}




void chat(Model * model, char * system_prompt){

	size_t max_tokens = (size_t)(model->config.sequence_maxtokens);

	alloc_tokens_batch(&input_tokens, &n_input_tokens, 1, max_tokens);

	printf("\r\n\n\n");

	
	int pos = 0;

	while( pos < max_tokens ){

		user_prompt[0] = '\0';


		if(chat_flags & CHAT_LOAD_CONTEXT){

			chat_load_context(model, input_tokens[0], &n_input_tokens[0]);
			pos = n_input_tokens[0];

			chat_flags ^= CHAT_LOAD_CONTEXT;
		}
		else{
	
	
			if(chat_scheme == CHATSCHEME_TINYLLAMA){
				strcpy(user_prompt, "<|user|>\n");
			}
			else
			if(chat_scheme == CHATSCHEME_LLAMA){
				strcpy(user_prompt, "[INST] ");
			}
			else
			if(chat_scheme == CHATSCHEME_GEMMA){
				if(pos==0)	input_tokens[0][n_input_tokens[0]++] = toki.bos_id; 
				input_tokens[0][n_input_tokens[0]++] = 106; //<start_of_turn>
				strcpy(user_prompt, "user");	
				strcpy(endstr(user_prompt), " ");	
			}
			else{
				//do nothing
			}
	
	
			if((pos==0) && (system_prompt!=NULL) && (strlen(system_prompt)!=0)){

				if(chat_scheme == CHATSCHEME_TINYLLAMA){
					//sprintf(user_prompt, "<|system|>\n%s</s> \n<|user|>\n", system_prompt);
	
					sprintf(user_prompt, "<|system|>\n%s", system_prompt);
	
					//we need to explicitly put EOS token after prompt:
					text2tokens(model, &toki, user_prompt, (0x1|0x2), input_tokens[0], &n_input_tokens[0]);
	
					input_tokens[0][n_input_tokens[0]++] = 29871; 
	
					//let's go on: 
					//we complete the tail of "system" turn with " \n", 
					//and then go on with the "user" turn
					strcpy(user_prompt, "\n<|user|>\n");
				}
				else
				if(chat_scheme == CHATSCHEME_LLAMA){
					sprintf(endstr(user_prompt), "<<SYS>>\n%s\n<</SYS>>\n\n", system_prompt);
				}
				else
				if(chat_scheme == CHATSCHEME_GEMMA){
					sprintf(endstr(user_prompt), "%s\n", system_prompt);
				}
				else{
					sprintf(user_prompt,"%s",system_prompt);
				}
				
	
			}
		}


		int put_BOS;


		if(chat_flags & CHAT_SAVE_CONTEXT){

			put_BOS = ((pos==0) && (chat_scheme!=CHATSCHEME_TINYLLAMA) && (chat_scheme!=CHATSCHEME_GEMMA)) ? 1 : 0;	
			//NOTE: also TinyLlama and Gemma requires BOS, but we'be already put it above
			text2tokens(model, &toki, user_prompt, ((put_BOS==1)?0x1:0x0), input_tokens[0], &n_input_tokens[0]); 


			//let's start the processing of the input context; we will then save the key cache and the value cache, and exit	
			pos = forward(model, NULL, NULL, input_tokens, n_input_tokens, (size_t)pos, ATTENTION_CAUSAL, 1, (size_t)(n_input_tokens[0]-pos));

			return;
		}



		printf("\r\nCARLO:  ");
		get_userprompt( endstr(user_prompt), (PROMPT_MAXLEN-strlen(user_prompt)-10) );		


		if(chat_scheme == CHATSCHEME_TINYLLAMA){
			//strcpy(endstr(user_prompt), "</s> \n");

			//we need to explicitly put EOS token after prompt:
			text2tokens(model, &toki, user_prompt, (0x0|0x2), input_tokens[0], &n_input_tokens[0]); 

			input_tokens[0][n_input_tokens[0]++] = 29871;

 
			//and then we put the tail to the "user" turn with " \n"
			strcpy(user_prompt, "\n");
			//and then we add the explicit head of the "assistant" role "<|assistant|>\n"
			strcpy(endstr(user_prompt), "<|assistant|>\n");
		}
		else
		if(chat_scheme == CHATSCHEME_LLAMA){
			strcpy(endstr(user_prompt), " [/INST]");
		}
		else
		if(chat_scheme == CHATSCHEME_GEMMA){
			text2tokens(model, &toki, user_prompt, (0x0|0x0), input_tokens[0], &n_input_tokens[0]); 
			input_tokens[0][n_input_tokens[0]++] = 107;	//<end_of_turn>
			strcpy(user_prompt, "\n");
			text2tokens(model, &toki, user_prompt, (0x0|0x0), input_tokens[0], &n_input_tokens[0]); 
			input_tokens[0][n_input_tokens[0]++] = 106;	//<start_of_turn>
			strcpy(user_prompt, "model");
		}
		else{
			//do nothing
		}


		//let's tokenize the current user prompt

		put_BOS = ((pos==0) && (chat_scheme!=CHATSCHEME_TINYLLAMA) && (chat_scheme!=CHATSCHEME_GEMMA) && (!(chat_flags & CHAT_LOAD_CONTEXT))) ? 1 : 0;	


		//NOTE: also TinyLlama and Gemma requires BOS, but we'be already put it above
		text2tokens(model, &toki, user_prompt, ((put_BOS==1)?0x1:0x0), input_tokens[0], &n_input_tokens[0]); 


		if(chat_flags & CHAT_LOAD_CONTEXT){

			chat_flags ^= CHAT_LOAD_CONTEXT;
		}



		//let's get the answer from TRiP!	
		printf("\r\n");
		printf("\r\nTRiP:  ");
		pos = forward(model, NULL, NULL, input_tokens, n_input_tokens, (size_t)pos, ATTENTION_CAUSAL, 1, (size_t)(n_input_tokens[0]-pos));
		
		printf("\r\n\n\n");
	}


	free_tokens_batch(&input_tokens, &n_input_tokens, 1, max_tokens);
	
}



