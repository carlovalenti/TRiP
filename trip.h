#ifndef TRIP_H
#define TRIP_H

// ============================================================
//
//  TRiP — Transformer Inference & Training Platform
//  A single-author, all-in-one C engine for transformer models.
//
//  Supports: Llama, Gemma, PaliGemma, GPT-2
//  Features: inference, training, chat, vision, tokenizer creation
//
//  by Carlo Valenti
//
// ============================================================
//
//  FILE MAP — Where to find what:
//
//  trip.h       YOU ARE HERE. Every type, struct, global, and
//               declaration lives here; this is the map.
//
//  math.c       The building blocks of a transformer, in a paired
//               forward+backward form: matmul, softmax, layernorm,
//               RMSnorm, RoPE, attention head I/O, FFN activations,
//               vector arithmetic. Each op is followed immediately
//               by its gradient — so that you can read them together.
//
//  forward.c    The forward pass: the single function that wires all
//               the math.c primitives into a full transformer layer
//               stack. Also: token sampling (temperature, top-p, top-k).
//
//  backward.c   The backward pass: the mirror of forward.c, walking
//               the layers in reverse to compute gradients. Then the
//               optimizer: AdamW with cosine annealing, gradient
//               clipping, and the full model update loop.
//
//  model.c      Everything about loading, saving, and managing a model
//               in memory: checkpoint formats (safetensors, Karpathy's
//               llama2.c/gpt2), weight initialization, memory allocation
//               for activations and gradients, tokenizer (BPE, build,
//               encode, decode), and vision preprocessing.
//
//  utils.c      Infrastructure: logging, a JSON parser (AI-written 90%),
//               markdown-aware terminal output, JPEG loading via
//               libjpeg-turbo and X11 image display (AI written 99%), 
//               and the management of terminal raw mode.
//
//  main.c       The front end: CLI argument parsing, the chat loop,
//               the training loop, the inference loop, the vision
//               pipeline orchestration. This is where it all starts.
//
// ============================================================

// ============================================================
//  Version coherence
//
//  Each .c file declares its own version as a literal BEFORE
//  including trip.h. trip.h checks it at compile time.
//  If you modify a file, bump its version here AND in the
//  file itself. Forget either one → the build fails.
//
//  Version format: YYYYMMDDVV (read as a date, compare as
//  a number). E.g. 2026-04-15, revision 01 → 2026041501
// ============================================================

#define	TRIP_VERSION_NUMBER_STRING	"V4.00"

#define TRIP_VERSION                    2026041501
#define TRIP_VERSION_STRING             "2026-04-15.01"

#define TRIP_MATH_VERSION_EXPECTED      2026041501
#define TRIP_FORWARD_VERSION_EXPECTED   2026041501
#define TRIP_BACKWARD_VERSION_EXPECTED  2026041501
#define TRIP_MODEL_VERSION_EXPECTED     2026041501
#define TRIP_UTILS_VERSION_EXPECTED     2026041501

#ifdef TRIP_MATH_VERSION
  #if TRIP_MATH_VERSION != TRIP_MATH_VERSION_EXPECTED
    #error "math.c version mismatch with trip.h"
  #endif
#endif

#ifdef TRIP_FORWARD_VERSION
  #if TRIP_FORWARD_VERSION != TRIP_FORWARD_VERSION_EXPECTED
    #error "forward.c version mismatch with trip.h"
  #endif
#endif

#ifdef TRIP_BACKWARD_VERSION
  #if TRIP_BACKWARD_VERSION != TRIP_BACKWARD_VERSION_EXPECTED
    #error "backward.c version mismatch with trip.h"
  #endif
#endif

#ifdef TRIP_MODEL_VERSION
  #if TRIP_MODEL_VERSION != TRIP_MODEL_VERSION_EXPECTED
    #error "model.c version mismatch with trip.h"
  #endif
#endif

#ifdef TRIP_UTILS_VERSION
  #if TRIP_UTILS_VERSION != TRIP_UTILS_VERSION_EXPECTED
    #error "utils.c version mismatch with trip.h"
  #endif
#endif


#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctype.h>
#include <stdint.h>
#include <errno.h>
#include <float.h>
#include <stdbool.h>
#include <termios.h>
#include <signal.h>
#include <omp.h>
#include <execinfo.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>


#if ( __GNUC__ > 13  ||  (__GNUC__ == 13 && __GNUC_MINOR__ >= 0) )
  #include <x86intrin.h>
#else
  typedef uint16_t __bf16;
#endif


//this is just for doing quick casts from void, no further special meaning
typedef uint8_t byte;


// ============================================================
//  Constants and configuration defines
// ============================================================

#define LLEN			4096
#define MAX_TEXT_LEN		((4096*1024)-1)

#define	PROMPT_MAXLEN	32768


#define	WTYPE_UNDEFINED	0
#define	WTYPE_BF16	1
#define	WTYPE_FLOAT32	2
#define	WTYPE_FLOAT16	3


#define	LOG_ERROR		1
#define LOG_INFO		2
#define	LOG_VERBOSE_INFO	3
#define	LOG_DEBUG		4
#define	LOG_VERBOSE_DEBUG	5


#define CHAT_SAVE_CONTEXT	0x1
#define CHAT_LOAD_CONTEXT	0x2


#define MAX_VOCAB_ENTRIES	(512*1024)	
#define MAX_TOKEN_LEN		100
#define MAX_INPUT_TOKENS	(128*1024)	


#define	PARALLEL_FORWARDING_OFF	0
#define	PARALLEL_FORWARDING_ON	1


#define	RAM_NO_OPTIMIZATIONS		0
#define	RAM_BASIC_OPTIMIZATIONS		1


#define	ACTION_NONE			0
#define	ACTION_DECODE			1
#define	ACTION_CHAT			2
#define	ACTION_VISION			3
#define	ACTION_TRAIN			4
#define	ACTION_CREATE			5
#define	ACTION_CREATE_VOCABULARY	6


#define	ARCH_BASIC			0
#define	ARCH_LLAMA_CAUSAL		1
#define	ARCH_GEMMA_CAUSAL		2
#define	ARCH_PALIGEMMA_CONDITIONAL	3
#define	ARCH_GPT2_CAUSAL		4


#define	MODELTYPE_UNDEFINED		0
#define	MODELTYPE_DECODER		1
#define	MODELTYPE_VISION_ENCODER	2


#define	CHATSCHEME_NONE			0
#define	CHATSCHEME_LLAMA		1
#define	CHATSCHEME_GEMMA		2
#define	CHATSCHEME_TINYLLAMA		3


#define	TOKFORMAT_TRIP			0
#define	TOKFORMAT_LLAMA2_AK		1
#define	TOKFORMAT_JSON_HUGGINGFACE	2
#define	TOKFORMAT_GPT2_AK		3


#define	TOKTYPE_TRIP		0
#define	TOKTYPE_SENTENCEPIECE	1


#define	CP_UNDEFINED	0
#define	CP_LLAMA2_AK	1
#define	CP_SAFETENSORS	2
#define	CP_GPT2_AK	3


#define	EMBEDDINGS_UNSHARED	0
#define	EMBEDDINGS_SHARED	1


#define	POSE_UNDEFINED	0
#define	POSE_ORIGINAL	1
#define	POSE_ROPE	2
#define	POSE_LEARNED	3


#define	NORM_NONE	0
#define	NORM_LAYERNORM	1
#define	NORM_RMSNORM	2

#define	GATE_OFF	0
#define	GATE_ON		1

#define	NL_RELU			1
#define	NL_GELU_SIGMOID		2
#define	NL_GELU_TANH		3
#define	NL_SILU_LLAMA		4

#define	BIAS_OFF		0
#define	BIAS_ON			1


//SAMPLING CONFIGURATION
#define	DEFAULT_TOP_P		0.9
#define	DEFAULT_TEMPERATURE	1.0


#define	ATTENTION_CAUSAL	1
#define	ATTENTION_FULL		2

#define	MEMORY_FORWARD		1
#define	MEMORY_GRADIENTS	2

#define	TRIPTYPE_INT		1
#define	TRIPTYPE_FLOAT		2
#define	TRIPTYPE_BOOL		3
#define	TRIPTYPE_STRING		4


#define ANSI_BOLD "\033[1;97m"  // Bold (1) with bright white (97)
#define ANSI_ITALIC "\033[3;38;5;39m"  // Italic + bright blue color
#define ANSI_CODE "\033[1;100m"  // Bold text with gray background
#define ANSI_RESET "\033[0m"




// ============================================================
//  Data structures
// ============================================================


typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {

    char ** vocab;
    float * vocab_scores;
    TokenIndex * sorted_vocab;
    int vocab_size;
    int max_token_length;
    int pad_id; 
    int bos_id; 
    int eos_id;
    int byte_fallback_offset;	//in the vocabulary, this is the id of the token containing byte 0x00, being the first single-byte token

    int singlebytes_space_firstid;
    int singlebytes_space_lastid;

    int addedtokens_space_firstid;
    int addedtokens_space_lastid;

} Tokenizer;


typedef struct Pixel {
	byte R;
	byte G;
	byte B;
} Pixel;

typedef struct Picture {
	Pixel * pic;
	int width;
	int height;
} Picture;


// JSON Node Types
typedef enum {
    JSON_NULL,
    JSON_STRING,
    JSON_NUMBER,
    JSON_BOOL,
    JSON_OBJECT,
    JSON_ARRAY,
    JSON_UNKNOWN
} JsonNodeType;

// JSON Node Structure
typedef struct JsonNode {
    JsonNodeType type;
    char *key;              // Key name for object members
    union {
        char *stringValue;  // For string values
        double numberValue; // For number values
        int boolValue;      // For boolean values
        struct JsonNode *children; // For objects and arrays (linked list)
    } value;
    struct JsonNode *next;  // Next sibling in linked list
} JsonNode;


typedef struct {

	int architectures;	//architecture type (will add special management for some architecture, e.g.: Google Gemma)
	int submodel_type;	//if the full model is composed by several sub-models (e.g.: encoder + decoder; maybe in future: MoE),
				//this field specifies if this sub-model is encoder or decoder. Not required if the model is not a composition of sub-models.

	int shared_weights_memory;	//this field specifies if this model is a sub-model which can share the same weights area (from safetensors files) with another sub-model


	//vision model extra configuration (encoder)
	int vision_image_tokens;		//this is "max_position_embeddings" of the vision model, being the number of patches
	int vision_patch_size;			//this is the length (in pixels) of the side of the square of pixels ("patch")
	int target_dim_stream;	//if the vision encoder outputs to a decoder, this is the size of stream of the decoder (required for the multimodal projector)


	//language model extra configuration (decoder)
	int vocab_size;		//it's a parameter of the model, more than of the tokenizer, because:
				//1) before the initial layer, the model translates the input token to the corresponding embedding (and this is a result of the training)
				//2) in the final layer, the model projects the calculated probability distribution over all the entries of the vocabulary
				//
	int sequence_maxtokens;	//maximum number of tokens in the sequence
				//
	int embeddings_cfg;	//configuration for token embeddings tying
				//
				//values:
				// "0":	EMBEDDINGS_UNSHARED	logit classifier has its own weights
				// "1":	EMBEDDINGS_SHARED	token embeddings and logit classifier share the same weights
				//



	//configuration for all model types (decoder or encoder)
	int dim_stream;		//"dimensionality of the model", i.e. the number of scalar elemnts in the stream vector (residual stream)
	int ffn_hidden_dim;	//dimension of the hidden layer of neurons in the feed-forward-network layers
	int n_layers;		//number of "attention + FFN" layers
	int n_queries;		//number of queries (query heads) at each layer
	int n_keys;		//number of keys (thus: values) at each layer; if this is a sub-multiplier of n_queries, then:
				//1) we are using a multi-query attention (n_keys=1)
				//   OR
				//2) we are using a grouped-query attention (n_keys sub-multiplier for n_queries)
	int pose_cfg;		//type of positional embeddings
	int norm_cfg[4];	//configuration of the normalizations:
				// [0]:	PRE-attention  normalization type
				// [1]: POST-attention normalization type
				// [2]: FINAL          normalization type
				// [3]: free for future usage
				//
				// values:
				// "0": NORM_NONE		no normalization
				// "1": NORM_LAYERNORM		layernorm
				// "2": NORM_RMSNORM 		rmsnorm
				//
	int ffn_nl_type[2];	//configuration for non-linearities in the Feed Forward Networks
				// [0]:	configuration for the post-attention FFN at each layer
				// [1]: 
				//
				// [0] possible values:
				// "0":	NOT USED, NOT ALLOWED
				// "1":	RELU
				// "2": GELU_SIGMOID
				// "3": GELU_TANH
				// "4": SILU_LLAMA
				//
				// [1] possible values:
				// "0": GATE_OFF	no extra gate_proj
				// "1": GATE_ON		the non-linearity is applied to gate_proj(input) instead that to up_proj(input), 
				// 			and then it's multiplied element-wise with up_proj(input)
				//
	int bias_cfg[4];	//configuration for the presence/absence of th biases in the linear layers right before and after the Feed Forward Network
				// [0]: configuration for the linear layer BEFORE the FFN
				// [1]: configuration for the linear layer AFTER  the FFN
				// [2]: UNUSED: configuration for the norms; currently it's unused, since each norm requires or not biases depending on the type of norm
				// [3]: configuration for the q-k-v-o projections
				//
				// values:
				// "0": BIAS_OFF	no biases, weights only
				// "1": BIAS_ON		biases are present

} Configuration;

typedef struct {

	//extra tensors for vision model (encoder)
	byte * vision_embeddings_w;	// flattened_patch_size * dim_stream	
	byte * vision_embeddings_b;	// dim_stream
	byte * learned_pose_w;		// n_patches * dim_stream

	byte * multimodal_projector_w;	// dim_stream * target_dim_stream
	byte * multimodal_projector_b;	// 1          * target_dim_stream
 

	//extra tensors for language model (decoder)
	byte * embeddings;		// vocab_size * dim_stream 
	byte * logits_classifier;	// dim_stream * vocab_size (translates the output of the final layer to the probability vector over all the possible vocabulary entries)


	//tensors for all model types (decoder or encoder)
	//1) per-layer tensors
	byte ** norm_pre_w;	// n_layers * dim_stream                       (output weights for PRE-attention normalization, if present)
	byte ** norm_pre_b;	// n_layers * dim_stream                       (output bias for PRE-attention normalization, if present, and if it requires biases)
	byte ** qm;		// dim_stream * dim_qkv * n_queries * n_layers (matrixes for calculating the QUERY  vectors)
	byte ** qb;		// 1          * dim_qkv * n_queries * n_layers (biases   for calculating the QUERY  vectors)
	byte ** km;		// dim_stream * dim_qkv * n_keys * n_layers    (matrixes for calcualting the KEY    vectors)
	byte ** kb;		// 1          * dim_qkv * n_keys * n_layers    (biases   for calcualting the KEY    vectors)
	byte ** vm;		// dim_stream * dim_qkv * n_keys * n_layers    (matrixes for calculating the VALUE  vectors)
	byte ** vb;		// 1          * dim_qkv * n_keys * n_layers    (biases   for calculating the VALUE  vectors)
	byte ** om;		// dim_stream * dim_qkv * n_queries * n_layers (matrixes for calculating the OUTPUT vectors)
	byte ** ob;		// 1          * dim_qkv * n_queries * n_layers (biases   for calculating the OUTPUT vectors)
	byte ** norm_post_w;	// n_layers * dim_stream                       (output weights for POST-attention normalization, if present)
	byte ** norm_post_b;	// n_layers * dim_stream                       (output bias for POST-attention normalization, if present, and if it requires biases)

	byte ** pre_ffn_w;	// n_layers * dim_stream * ffn_hidden_dim      (weights for the PRE-FeedForwardNetwork  linear layer)
	byte ** pre_ffn_b;	// n_layers * 1          * ffn_hidden_dim      (biases  for the PRE-FeedForwardNetwork  linear layer, if it requires biases)
	byte ** pre_ffn_w2;	// n_layers * dim_stream * ffn_hidden_dim      (secondary weights for the PRE-FeedForwardNetwork  linear layer, as it is required by LLAMA2 architecture)
	byte ** post_ffn_w;	// n_layers * ffn_hidden_dim * dim_stream      (weights for the POST-FeedForwardNetwork linear layer)
	byte ** post_ffn_b;	// n_layers * 1              * dim_stream      (biases  for the POST-FeedForwardNetwork linear layer, if it requires biases)
	//2) single-layer	tensors
	byte * norm_final_w;	// 1 * dim_stream (output weights for FINAL normalization)
	byte * norm_final_b;	// 1 * dim_stream (output bias for FINAL normalization, if it requires biases)
				
} Weights;


typedef struct {

        //runstate vectors (always in float32, design choice; I keep them from the "byte **" type to be ready for any change)

        byte ** residualstream_layerstart;

        byte ** norm_pre_stream;


	byte ** queries;

 	byte ** keys;
	byte ** values;


        byte ** raw_attention_scores;
        byte ** attention_scores;

        byte ** heads_output;

        byte ** attentionlayer_out_stream;

	byte ** residualstream_after_attention; 

        byte ** norm_post_stream;
        byte ** ffn_in_stream;
        byte ** ffn_aux_stream;
        byte ** ffn_out_stream;
        byte ** ffn_final_stream;

	byte ** residualstream_after_ffn; 

        byte * logits;

	
	//extra vectors, used only when training, during the forward pass
 
	byte * norm_final_stream;

	byte * norm_final_mean;
	byte * norm_final_rstd;
	byte * norm_final_rrms;


	byte ** norm_post_mean;
	byte ** norm_post_rstd;
	byte ** norm_post_rrms;


	byte ** norm_pre_mean;
	byte ** norm_pre_rstd;
	byte ** norm_pre_rrms;


} ForwardMemory;


typedef struct {

        byte * dlogits;
        byte * dlogits_classifier;
        byte * dnorm_final_stream;
	byte * dnorm_final_w;
	byte * dnorm_final_b;


	byte ** dresidualstream_after_ffn;

	byte ** dffn_in_stream;	
	byte ** dffn_aux_stream;	
	byte ** dffn_out_stream;	
	byte ** dffn_final_stream;	
	byte ** dpre_ffn_w;	
	byte ** dpre_ffn_b;	
	byte ** dpre_ffn_w2;	
	byte ** dpost_ffn_w;	
	byte ** dpost_ffn_b;	

	byte ** dnorm_post_stream;

	byte ** dnorm_post_w;
	byte ** dnorm_post_b;

	byte ** dresidualstream_after_attention;

	byte ** dattentionlayer_out_stream;	

	byte ** dom;	
	byte ** dob;	
	byte ** dheads_output;	

	byte ** dattention_scores;	
	byte ** draw_attention_scores;
	
	byte ** dqueries;	
	byte ** dkeys;	
	byte ** dvalues;	

	byte ** dqm;	
	byte ** dqb;	
	byte ** dkm;	
	byte ** dkb;	
	byte ** dvm;	
	byte ** dvb;	
	

	byte ** dnorm_pre_stream;

	byte ** dnorm_pre_w;
	byte ** dnorm_pre_b;


	byte ** dresidualstream_layerstart;


	byte * dembeddings;

	byte * dlearned_pose_w;

} GradientsMemory;



typedef struct {

	Configuration config;
	Weights w;

	ForwardMemory fm;
	GradientsMemory grads;

	char * dirpath;			//directory path of the model files
	int nfiles;			//number of files the checkpoint is split into
	int * fd;			//array of file descriptors of the checkpoint files, if loaded/memory-mapped (mmap)
	unsigned char ** file_data;	//array of pointers to the checkpoint files raw data areas (if loaded)
	unsigned char ** tensors_data;	//array of pointers to the tensors data areas (if loaded)
	unsigned char ** header_data;	//array of pointers to the header data areas (if loaded)
	ssize_t * checkpoint_size;	//array of sizes of the checkpoint files

} Model;


typedef struct adamw_mv {
	byte * params;	//the address of the weights is stored as label/key for lookup when we will need to update these weights
	float * m;
	float * v;
	struct adamw_mv * next;
} adamw_mv;


struct adamw_cfg {
	float learning_rate;
	float beta1;
	float beta2;
	float epsilon;
	float weight_decay;

	size_t step;

	struct adamw_mv * root;
};


typedef struct {
	float prob;
	int index;
} LogitElement;




// ============================================================
//  Global variables (defined in main.c)
// ============================================================

extern char lbuf[];
extern char inbuf[];

extern char user_prompt[];
extern char vision_picture_path[];

extern const int wtype_bytesize[];
extern char * wtype_text[];

extern int wtype;
extern size_t wsize;

extern int target_wtype;
extern size_t target_wsize;

extern float norm_eps;
extern float pose_theta;

extern int log_cfg;
extern bool calculate_loss;
extern int runtime_actions;
extern int build_vocab_size;
extern char * input_text_path;

extern char * tok_path;
extern char * mod_path;
extern char * cfg_path;

extern char * train_cfg_path;
extern char * train_data_path;

extern ssize_t training_batch_size;
extern ssize_t training_max_steps;
extern ssize_t training_num_epochs;
extern ssize_t training_save_steps;

extern char training_lr_scheduler_type[32];
extern ssize_t training_warmup_steps;
extern float training_min_lr;

extern char * chat_context_file;
extern int chat_flags;

extern int vision_detect_status;
extern int vision_detect_y_min;
extern int vision_detect_x_min;
extern int vision_detect_y_max;
extern int vision_detect_x_max;

extern const char * log_label[];

extern int ** input_tokens;
extern int *  n_input_tokens;

extern int parallel_forwarding;
extern int ramflag;

extern const char * action_text[];
extern int action;
extern const char * arch_text[];
extern const char * modeltype_text[];
extern const char * chatscheme_text[];
extern int chat_scheme;
extern const char * tokformat_text[];
extern int tokenizer_format;
extern const char * toktype_text[];
extern int tokenizer_type;
extern const char * cp_litteral[];
extern int checkpoint_type;
extern const char * embeddings_cfg_text[];
extern const char * pose_cfg_text[];

extern float * rope_subk;
extern size_t rope_subk_timesteps;
extern size_t rope_lastpos;

extern const char * norm_cfg_text[];
extern const char * ffn_nl_type_text[];
extern const char * bias_cfg_text[];

extern float temperature;
extern float top_p;
extern int top_k;

extern Tokenizer toki;

extern char num2str_buf[];

extern bool chat_text_bold;
extern bool chat_text_italic;
extern bool chat_text_code;

extern Picture * vision_picture;

extern const char * jsontype_text[];

extern long int start_ts;
extern long int last_ts;

extern char * json_multilabels[];

extern struct adamw_cfg adamw_cfg;

extern const char banner[];



// ============================================================
//  Function declarations — math.c
// ============================================================

// --- Attention ---
int attention_head_io(Model * model, byte * _raw_attention_scores, byte * _attention_scores, byte * _head_out, byte * _queries, size_t q_startpos, byte * _keys, byte * _values, size_t kv_startpos, size_t kv_npos, int attention_type, size_t B, size_t T, int ** intok);
int attention_head_io_backward(
	byte * _dqueries, byte * _dkeys, byte * _dvalues,
	byte * _queries, byte * _keys, byte * _values, size_t q_startpos, size_t kv_startpos, size_t kv_npos, byte * _dhead_out,
	byte * _attention_scores, byte * _raw_attention_scores, byte * _dattention_scores, byte * _draw_attention_scores, 
	Model * model, int attention_type, size_t B, size_t T
);

// --- Positional embeddings ---
int apply_positional_embeddings(Model * model, byte * _out_vector, byte * _in_vector, size_t dim_vector, size_t pos_offset, size_t B, size_t T);
int positional_embeddings_backward(
	byte * _din, byte * _dlearned, byte * _dout,
	Model * model, size_t dim_vector, size_t pos_offset, size_t B, size_t T
);
int token_embeddings_backward(
	byte * _din, byte * _dout, int ** intok,
	Model * model, size_t B, size_t T
);

// --- Softmax & loss ---
void softmax(byte * _a, size_t ax, size_t ay, byte * _b);
void softmax_backward(byte * _din, byte * _dout, byte * _out, size_t ax, size_t ay);
void crossentropy(float * calc_prob, float * target_prob, size_t size, float * loss);
void crossentropy_softmax_backward(
	byte * _dlogits, byte * _probs, int ** intok,
	Model * model, int B, int T
);

// --- Vector ops ---
void multiply_vector(byte * _va, size_t dim, float factor, byte * _vout);
void multiply_vector_backward(byte * _dva, byte * _dfactor, byte * _va, size_t dim, float factor, byte * _dvout);
void sum_vectors(byte * _va, int va_wtype, byte * _vb, int vb_wtype, size_t dim, byte * _vout);
void sum_vectors_backward(byte * _da, byte * _db, byte * _dout, size_t dim);

// --- Matrix multiply ---
int matmulf_nt(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c);
int matmulf_nt_backward(byte * _da, byte * _db, byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _dc, size_t B, size_t T);
int matmulf_nt_interleaved(byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _c);
int matmulf_nt_interleaved_backward(byte * _da, byte * _db, byte * _a, byte * _b, size_t ax, size_t ay, size_t bx, size_t by, byte * _dc, size_t B, size_t T);

// --- Normalization ---
int layernorm(byte * _out, byte * _in, byte * _output_weights, byte * _output_bias, size_t dim, size_t B, size_t T, byte * _mean, byte * _rstd);
int layernorm_backward(byte * _din, byte * _dweight, byte * _dbias, byte * _dout, byte * _in, byte * _output_weights, byte * _mean, byte * _rstd, size_t B, size_t T, size_t dim);
int rmsnorm(Model * model, byte * _out, byte * _in, byte * _output_weights, size_t dim, size_t B, size_t T, byte * _rrms);
int rmsnorm_backward(byte * _din, byte * _dweight, byte * _dout, byte * _in, byte * _output_weights, byte * _rrms, Model * model, size_t B, size_t T, size_t dim);

// --- FFN activation ---
int ffn_io(int nl_type, byte * _out, byte * _in, byte * _in2, size_t io_size);
int ffn_io_backward(byte * _din, byte * _din2, byte * _dout, byte *_in, byte *_in2, int nl_type, size_t io_size);



// ============================================================
//  Function declarations — forward.c
// ============================================================

int forward(Model * model, byte * embeddings, byte * out_streams, int ** intok, int * n_intoks, size_t pos, int attention_type, size_t B, size_t T);
int sample_next_token(Model * model, byte * _logits, float * out_prob);
int compare_logit_elements(const void * aa, const void * bb);
void wdebug(byte * _in, int wtype_in, size_t dim, char * label, ssize_t index, int terminate);



// ============================================================
//  Function declarations — backward.c
// ============================================================

int backward(Model * model, int attention_type, int ** intok, size_t B, size_t T);
int adamw_set_config(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay);
int adamw_init();
adamw_mv * adamw_create_mv(byte * params, size_t size);
int adamw_free();
int adamw(byte * params, byte * grads, size_t size);
int adamw_set_step(size_t step);
float cosine_annealing_lr(ssize_t step, ssize_t warmup_steps, ssize_t total_steps, float max_lr, float min_lr);
float tensorgrad_norm_squared(byte * _params, byte * _grads, size_t size);
bool gradients_multiply(float k, byte * _params, byte * _grads, size_t size);
bool gradients_check(Model * model, size_t T);
bool model_update(Model * model, size_t step, size_t T);



// ============================================================
//  Function declarations — model.c
// ============================================================

// --- Safetensors / checkpoint ---
byte * safetensors_get_tensor(Model * model, char * key_label, int layer, char * safetensors_prefix);
size_t get_checkpointfile_headersize(Model * model, int whichfile);
void load_model_from_checkpoint(Model * model, char * path);
void add_tensor_metadata(const char * name, size_t * shape, int n_dims, size_t * offset, char * json_buffer, char * dtype_str);
size_t convert_and_write_tensor(FILE* f, byte* src_data, size_t num_elements);
size_t convert_and_write_tensor_interleaved(FILE* f, byte* src_data, size_t num_elements, size_t row_size);
int save_model(Model * model);

// --- Model init / config ---
ssize_t init_model(Model * model, int checkpoint_type, Model * brother_model);
void check_model_configuration(Model * model);
void print_model_configuration(Model * model);
void load_training_args(char * filename);
void save_loss(float * sequence_loss, int * n_intoks, size_t B);

// --- Weight init ---
float random_normal();
void initialize_tensor_normal(float * tensor, size_t num_elements, float mean, float stddev);
void initialize_tensor_xavier(float * tensor, size_t num_elements, size_t fan_in, size_t fan_out);
void initialize_tensor_he(float * tensor, size_t num_elements, size_t fan_in);
void initialize_tensor_constant(float * tensor, size_t num_elements, float value);
void init_weights(Model * model);

// --- Memory management ---
int alloc_forward_memory(Model * model, size_t B, size_t T, int action);
void free_forward_memory(Model * model);
int free_mem(Model * model, void * addr, ssize_t layer, int memory);
int alloc_gradients_memory(Model * model, size_t B, size_t T);
int free_gradients_memory(Model * model, size_t B, size_t T);
int alloc_kv_memory(Model * model, size_t B, size_t T, size_t this_action);
void free_kv_memory(Model * model);
void free_model_memory(Model * model);

// --- Tokenizer ---
void build_tokenizer(Model * model, Tokenizer * t, char * tokenizer_path, int vocab_size);
void save_tokenizer_json(Tokenizer * t, char * filepath, int vocab_size);
void save_tokenizer(Tokenizer * t, char * tokenizer_path, int vocab_size);
void escape_json_string(char * input, char * output);
int pretokenizer(Model * model, char * str, Tokenizer * t);
int str_lookup(Model * model, char * str, Tokenizer * t);
void text2tokens(Model * model, Tokenizer * t, char * input, unsigned int mode, int * tokens, int * n_tokens);
int add_vocab_entry(Tokenizer * t, char * text, int * size);
int is_separator(char * text);
void dump_vocab(Tokenizer * t);
void create_vocab(Tokenizer * t, char * input_path);
int compare_tokens(const void * a, const void * b);
void mymemcpy(byte * out, byte * in, size_t len);

// --- Vision ---
byte * picture2streams(Model * model, Picture * picture);

// --- JSON config parser helpers ---
JsonNode * scanJson_multilabel(JsonNode * root, char * key_label, char * prefix, int layer);
int jsonstring_to_tripint(char * in);
float jsonstring_to_tripfloat(char * in);
JsonNode * json_get(JsonNode * root, char * file_text, char * TRIP_label, int TRIP_type, void * TRIP_target, int layer, char * default_value);



// ============================================================
//  Function declarations — utils.c
// ============================================================

void * myalloc(size_t nbytes);
char * int2str(int val);
char * float2str(float val);
char * endstr(char * inbuf);
bool is_hex_byte(const char* str, unsigned char* value);
void chat_textformat_reset();
void md_printf(const char* format, ...);
void mylog(int log_level, const char * format, ...);
void hexlog(char * header, byte * in, int len);
void print_stacktrace();
long int get_milliseconds();
void print_sampler_configuration();

// JSON parser
char *trimWhitespace(char *str);
JsonNode *createJsonNode(JsonNodeType type);
JsonNode *parseJsonString(char **json);
JsonNode *parseJsonNumber(char **json);
JsonNode *parseJsonNumber_as_String(char **json);
JsonNode *parseJsonBoolOrNull(char **json);
JsonNode *parseJsonArray(char **json);
JsonNode *parseJsonObject(char **json);
JsonNode *parseJsonValue(char **json);
void freeJsonTree(JsonNode *node);
void printJsonTree(JsonNode *node, int indent);
void printJsonNode(JsonNode *node, int indent);
JsonNode *findJsonNodeByKey(JsonNode *root, const char *key);

// Terminal
void enableRawMode(struct termios *orig_termios);
void disableRawMode(struct termios *orig_termios);
void get_userprompt(char * outbuf, int maxlen);

// Image / display
int XPutPixel(XImage *ximage, int x, int y, unsigned long pixel);
int XDestroyImage(XImage *ximage);
unsigned char* read_jpeg_to_rgb(const char* filename, unsigned int* width, unsigned int* height);
unsigned char* resize_rgb_buffer(unsigned char* input_buffer, unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height);
unsigned char* resize_rgb_lanczos(const unsigned char* input_buffer, unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height);
unsigned char* crop_rgb_buffer(unsigned char* input_buffer, unsigned int input_width, unsigned int input_height, unsigned int x, unsigned int y, unsigned int crop_width, unsigned int crop_height);
void displayPicture(Picture* pic, int timeout);
void displayPicture_resize(Picture * pic, int width, int height, int timeout);
void draw_rectangle(Picture* picture, int y_min, int x_min, int y_max, int x_max, byte R, byte G, byte B, int thickness, float fill_alpha);



// ============================================================
//  Function declarations — main.c
// ============================================================

void do_unit_tests();
void print_help();
void alloc_tokens_batch(int *** _input_tokens, int ** _n_input_tokens, size_t B, size_t T);
void free_tokens_batch(int *** _input_tokens, int ** _n_input_tokens, size_t B, size_t T);
void build_vocab(char * input_text_file, int size, char * input_tokenizer);
void check_platform(int action);
void signal_handler(int sig);
void chat_save_context(Model * model, int * intok, int * n_intoks);
void chat_load_context(Model * model, int * intok, int * n_intoks);
void chat(Model * model, char * system_prompt);



#endif // TRIP_H
