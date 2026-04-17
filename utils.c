#define TRIP_UTILS_VERSION 2026041501
#include "trip.h"

void * myalloc(size_t nbytes){
	void * pointer;
	if(posix_memalign( (void **)&pointer, CACHE_LINESIZE, nbytes) != 0){
		fprintf(stderr, "ERROR: posix_memalign failed for %zu bytes. Exiting...\n", nbytes);
		exit(-1);
	}
	for(size_t i = 0; i < nbytes; i++){
		((byte *)pointer)[i] = 0x00;
	}
	return pointer;
}




char * int2str(int val){
	sprintf(num2str_buf,"%d",val);
	return num2str_buf;
}
char * float2str(float val){
	sprintf(num2str_buf,"%f",val);
	return num2str_buf;
}

char * endstr(char * inbuf){
	return (inbuf + strlen(inbuf));
}



// ANSI escape codes for formatting


// Check if string starts with "<0x" and ends with ">" and contains valid hex
bool is_hex_byte(const char* str, unsigned char* value) {
    if (strlen(str) != 6) return false;  // "<0xHH>"
    if (str[0] != '<' || str[1] != '0' || 
        str[2] != 'x' || str[5] != '>') return false;
    
    char hex[3] = {str[3], str[4], '\0'};
    char* endptr;
    *value = (unsigned char)strtol(hex, &endptr, 16);
    return *endptr == '\0' && isxdigit(str[3]) && isxdigit(str[4]);
}



void chat_textformat_reset(){

	bool chat_text_bold = false;
	bool chat_text_italic = false;
	bool chat_text_code = false;

	printf(ANSI_RESET);
}


void md_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    // First, format the string with variable arguments
    char* formatted = malloc(1024);  // Adjust size as needed
    vsnprintf(formatted, 1024, format, args);
    
   char* ptr = formatted;
    
    while (*ptr) {
        // Handle hex byte values (<0xHH>)
        if (ptr[0] == '<' && strlen(ptr) >= 6) {
            unsigned char byte_val;
            char temp[7];
            strncpy(temp, ptr, 6);
            temp[6] = '\0';
            
            if (is_hex_byte(temp, &byte_val)) {
                putchar(byte_val);
                ptr += 6;
                continue;
            }
        }
	else
	if(ptr[0]=='\\'){
		switch(ptr[1]){
		case 'n':	printf("\n");	break;
		case 'r':	printf("\r");	break;
		case 't':	printf("\t");	break;
		default:
			if(ptr[1]=='\0'){
				ptr++;
				continue;
			}
			else{
				printf("%c",ptr[1]);	break;
			}
		}

		ptr += 2;
		continue;
	}
	else
	if((strlen(ptr)>=3) && (memcmp(ptr,"\xE2\x96\x81",3)==0)){
		printf(" ");
		ptr += 3;
		continue;
	}

        
        // Handle bold (**text**)
        if (ptr[0] == '*' && ptr[1] == '*') {
            if (!chat_text_bold) {
                printf(ANSI_BOLD);
                chat_text_bold = true;
            } else {
                printf(ANSI_RESET);
                chat_text_bold = false;
            }
            ptr += 2;
            continue;
        }
        
        // Handle italic (*text*)
        if (ptr[0] == '*' && ptr[1] != '*') {
            if (!chat_text_italic) {
                printf(ANSI_ITALIC);
                chat_text_italic = true;
            } else {
                printf(ANSI_RESET);
                chat_text_italic = false;
            }
            ptr++;
            continue;
        }
        
        // Handle inline code (`text`)
        if (ptr[0] == '`') {
            if (!chat_text_code) {
                printf(ANSI_CODE);  // Using bold text with gray background for code
                chat_text_code = true;
            } else {
                printf(ANSI_RESET);
                chat_text_code = false;
            }
            ptr++;
            continue;
        }
        
        // Print regular character
        putchar(*ptr);
        ptr++;
    }

    
    free(formatted);
    va_end(args);
}


void mylog(int log_level, const char * format, ...){

	if(log_level > log_cfg)	return;

	// Get current date and time
	time_t rawtime;
	struct tm * timeinfo;
	char time_str[32];
	
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);
	
	
	//FILE * f = fopen("log.txt","a");
	FILE * f = stdout;
	
	if(f == NULL){
	    perror("Failed to open log file");
	    return;
	}
	
	
	// Print date and time to log file
	fprintf(f, "%s - %s", time_str, log_label[log_level]);
	
	// Handle variable arguments
	va_list args;
	va_start(args, format);
	vfprintf(f, format, args);
	va_end(args);
	
	fprintf(f, "\n");
	
	if((f!=stdout) && (f!=stderr))	fclose(f);
}


void hexlog(char * header, byte * in, int len){
	char * out = malloc(strlen(header)+(len*3)+1);
	strcpy(out,header);
	for(int i=0;i<len;i++){
		sprintf(endstr(out)," %02X",in[i]);
	}
	mylog(LOG_INFO, out);
	free(out);
}


void print_stacktrace() {
    void * array[10];
    size_t size;
    char ** strings;
    
    size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    
    printf("Stack trace (%zd frames):\n", size);
    for(size_t i = 0; i < size; i++){
        printf("  %s\n", strings[i]);
    }
    
    free(strings);
}






long int get_milliseconds(){
	struct timespec t;
	long int ms;

	clock_gettime(CLOCK_REALTIME, &t);
	ms = (t.tv_sec*1000) + (t.tv_nsec/1000000);

	return ms;
}


void print_sampler_configuration(){

	mylog(LOG_INFO, "temperature = %.3f", temperature);

	if(temperature == 0.0){
		mylog(LOG_INFO, "  ==>  forcing deterministic greedy sampling (will always return the token with highest probability)");
	}
	else{
		if(top_k > 0){
			mylog(LOG_INFO, "top_k = %d   (  ==>  using top_k  )", top_k);
		}
		else{
			mylog(LOG_INFO, "top_p = %.3f", top_p);
		}
	}
}


// ============================================================
//  JSON parser
// ============================================================

char *trimWhitespace(char *str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;


//Carlo: may cause problem when the json ends with right brace just before the beginning of the tensors space in safetensors file
////    *(end + 1) = '\0';


    return str;
}

// Function to create a new JSON node
JsonNode *createJsonNode(JsonNodeType type) {
    JsonNode *node = (JsonNode *)malloc(sizeof(JsonNode));
    node->type = type;
    node->key = NULL;
    node->next = NULL;
    if (type == JSON_OBJECT || type == JSON_ARRAY) {
        node->value.children = NULL;
    } else {
        node->value.stringValue = NULL;
    }
    return node;
}

// Function to parse a JSON string value with support for single and double quotes
JsonNode *parseJsonString(char **json) {
    JsonNode *node = createJsonNode(JSON_STRING);
    char quoteChar = **json; // This can be either ' or "
    (*json)++; // Skip the opening quote
    char *start = *json;
    while((**json != quoteChar) && (**json != '\0')){
	if(**json == '\\'){
		(*json)++;
		if(**json == '\0')	break;
	}
	(*json)++;
    }
    size_t length = *json - start;
    node->value.stringValue = (char *)malloc(length + 1);
    strncpy(node->value.stringValue, start, length);
    node->value.stringValue[length] = '\0';
    (*json)++; // Skip the closing quote
    return node;
}

// Function to parse a JSON number value; 
// Carlo: I had to add 'e' and 'E' to properly handle scientific notation - very common among "epsilons"
JsonNode *parseJsonNumber(char **json) {
    JsonNode *node = createJsonNode(JSON_NUMBER);
    char *start = *json;
    while((isdigit(**json)) || (**json=='.') || (**json=='-') || (**json=='e') || (**json=='E')) (*json)++;
    char temp = **json;
    **json = '\0';
    node->value.numberValue = atof(start);
    **json = temp;
    return node;
}

//Carlo: function to parse a JSON number value, and store it as a string
//this is because very high numbers are not properly stored by the float32 format (like: tensors indexes in the safetensors format)
JsonNode *parseJsonNumber_as_String(char **json) {
    JsonNode *node = createJsonNode(JSON_STRING);
    char *start = *json;
    while((isdigit(**json)) || (**json=='.') || (**json=='-') || (**json=='e') || (**json=='E')) (*json)++;
    size_t length = *json - start;
    node->value.stringValue = (char *)malloc(length + 1);
    strncpy(node->value.stringValue, start, length);
    node->value.stringValue[length] = '\0';
    return node;
}



// Function to parse a JSON boolean or null value
JsonNode *parseJsonBoolOrNull(char **json) {
    JsonNode *node;
    if (strncmp(*json, "true", 4) == 0) {
        node = createJsonNode(JSON_BOOL);
        node->value.boolValue = 1;
        *json += 4;
    } else if (strncmp(*json, "false", 5) == 0) {
        node = createJsonNode(JSON_BOOL);
        node->value.boolValue = 0;
        *json += 5;
    } else if (strncmp(*json, "null", 4) == 0) {
        node = createJsonNode(JSON_NULL);
        *json += 4;
    } else {
        node = createJsonNode(JSON_UNKNOWN);
    }
    return node;
}

// Function to parse a JSON array value
JsonNode *parseJsonArray(char **json) {
    JsonNode *node = createJsonNode(JSON_ARRAY);
    (*json)++; // Skip the opening bracket
    JsonNode *child = NULL;
    JsonNode **current = &(node->value.children);
    *json = trimWhitespace(*json);

    while (**json != ']' && **json != '\0') {
        *current = parseJsonValue(json);
        current = &((*current)->next);
        *json = trimWhitespace(*json);
        if (**json == ',') (*json)++;
        *json = trimWhitespace(*json);
    }
    (*json)++; // Skip the closing bracket
    return node;
}

// Function to parse a JSON object value
JsonNode *parseJsonObject(char **json) {
    JsonNode *node = createJsonNode(JSON_OBJECT);
    (*json)++; // Skip the opening brace
    JsonNode *child = NULL;
    JsonNode **current = &(node->value.children);
    *json = trimWhitespace(*json);

    while (**json != '}' && **json != '\0') {
        // Handle both single and double quoted keys
        if (**json == '\"' || **json == '\'') {
            JsonNode *keyNode = parseJsonString(json);
            *json = trimWhitespace(*json);
            if (**json == ':') (*json)++;
            *json = trimWhitespace(*json);
            *current = parseJsonValue(json);
            (*current)->key = keyNode->value.stringValue;
            free(keyNode);
            current = &((*current)->next);
            *json = trimWhitespace(*json);
            if (**json == ',') (*json)++;
            *json = trimWhitespace(*json);
        } else {
            break; // Syntax error if not a valid key
        }
    }
    (*json)++; // Skip the closing brace
    return node;
}

// Function to parse a JSON value
JsonNode *parseJsonValue(char **json) {
    *json = trimWhitespace(*json);
    switch (**json) {
        case '\"':
        case '\'':
            return parseJsonString(json);
        case '-':
        case '+':
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            return parseJsonNumber_as_String(json);
        case 't':
        case 'f':
        case 'n':
            return parseJsonBoolOrNull(json);
        case '{':
            return parseJsonObject(json);
        case '[':
            return parseJsonArray(json);
        default:
            return createJsonNode(JSON_UNKNOWN);
    }
}

// Function to free a JSON tree
void freeJsonTree(JsonNode *node) {
    if (node == NULL) return;
    if (node->key != NULL) free(node->key);
    switch (node->type) {
        case JSON_STRING:
            free(node->value.stringValue);
            break;
        case JSON_OBJECT:
        case JSON_ARRAY:
            freeJsonTree(node->value.children);
            break;
        default:
            break;
    }
    freeJsonTree(node->next);
    free(node);
}


// Function to print the JSON tree (for testing)
void printJsonTree(JsonNode *node, int indent) {
    if (node == NULL) return;
    for (int i = 0; i < indent; i++) printf("  ");
    if (node->key) printf("\"%s\": ", node->key);
    switch (node->type) {
        case JSON_STRING:
            printf("\"%s\"\n", node->value.stringValue);
            break;
        case JSON_NUMBER:
            printf("%f\n", node->value.numberValue);
            break;
        case JSON_BOOL:
            printf("%s\n", node->value.boolValue ? "true" : "false");
            break;
        case JSON_NULL:
            printf("null\n");
            break;
        case JSON_OBJECT:
            printf("{\n");
            printJsonTree(node->value.children, indent + 1);
            for (int i = 0; i < indent; i++) printf("  ");
            printf("}\n");
            break;
        case JSON_ARRAY:
            printf("[\n");
            printJsonTree(node->value.children, indent + 1);
            for (int i = 0; i < indent; i++) printf("  ");
            printf("]\n");
            break;
        default:
            printf("Unknown type\n");
    }
    printJsonTree(node->next, indent);
}
//Carlo: function to print just the JSON node (and its children nodes)
void printJsonNode(JsonNode *node, int indent) {
    if (node == NULL) return;
    for (int i = 0; i < indent; i++) printf("  ");
    if (node->key) printf("\"%s\": ", node->key);
    switch (node->type) {
        case JSON_STRING:
            printf("\"%s\"\n", node->value.stringValue);
            break;
        case JSON_NUMBER:
            printf("%f\n", node->value.numberValue);
            break;
        case JSON_BOOL:
            printf("%s\n", node->value.boolValue ? "true" : "false");
            break;
        case JSON_NULL:
            printf("null\n");
            break;
        case JSON_OBJECT:
            printf("{\n");
            printJsonTree(node->value.children, indent + 1);
            for (int i = 0; i < indent; i++) printf("  ");
            printf("}\n");
            break;
        case JSON_ARRAY:
            printf("[\n");
            printJsonTree(node->value.children, indent + 1);
            for (int i = 0; i < indent; i++) printf("  ");
            printf("]\n");
            break;
        default:
            printf("Unknown type\n");
    }
//    printJsonTree(node->next, indent);
}


// Function to find a JSON node by key
JsonNode *findJsonNodeByKey(JsonNode *root, const char *key) {
    if (root == NULL) return NULL;

    JsonNode *current = root;
    while (current != NULL) {
        if (current->key && strcmp(current->key, key) == 0) {
            return current;
        }

        // Recursively search in child nodes if current node is an object or array
        if (current->type == JSON_OBJECT || current->type == JSON_ARRAY) {
            JsonNode *found = findJsonNodeByKey(current->value.children, key);
            if (found != NULL) {
                return found;
            }
        }

        current = current->next;
    }

    return NULL;
}


// ============================================================
//  Terminal I/O
// ============================================================



// Function to configure terminal settings
void enableRawMode(struct termios *orig_termios) {
    struct termios raw;
    
    // Save original terminal settings
    tcgetattr(STDIN_FILENO, orig_termios);
    raw = *orig_termios;
    
    // Modify settings for raw mode
    raw.c_lflag &= ~(ECHO | ICANON);
    
    // Apply new settings
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// Function to restore original terminal settings
void disableRawMode(struct termios *orig_termios) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, orig_termios);
}


void get_userprompt(char * outbuf, int maxlen){
    struct termios orig_termios;
    
    // Enable raw mode
    enableRawMode(&orig_termios);
    
    char * buffer = malloc(maxlen+1);
    int position = 0;
    char c;
    
    // Clear buffer
    memset(buffer, 0, maxlen+1);
    
    while (1) {
        c = getchar();
        
        if (c == '\n') {
            printf("\n");
            break;
        } else if(((c==127)||(c==8)) && (position > 0)) {
            position--;
            buffer[position] = '\0';
            printf("\b \b");  // Move back, print space, move back again
        } else if(((c!=127)&&(c!=8)) && (position < maxlen)) {
            buffer[position] = c;
            position++;
            printf("%c", c);  // Echo the character
        }
    }
    
    buffer[position] = '\0';
    strcpy(outbuf,buffer);
    free(buffer);  
 
    // Restore original terminal settings
    disableRawMode(&orig_termios);
}



//// libjpeg-turbo interface functions


// ============================================================
//  Image loading and display (libjpeg + X11)
// ============================================================

//// libjpeg-turbo interface functions


// Structure to handle JPEG errors
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct my_error_mgr* my_error_ptr;

// Custom error handler
METHODDEF(void) my_error_exit(j_common_ptr cinfo) {
    my_error_ptr myerr = (my_error_ptr)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

// Function to read JPEG into RGB buffer
unsigned char* read_jpeg_to_rgb(const char* filename, unsigned int* width, unsigned int* height){

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    FILE * infile;
    unsigned char * buffer = NULL;
    JSAMPARRAY scanlines = NULL;
    unsigned char * rowptr;

    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    // Set up error handler
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        free(buffer);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    // Set parameters for decompression
    cinfo.out_color_space = JCS_RGB;  // Force RGB output
    cinfo.scale_num = 1;              // No scaling by default
    cinfo.scale_denom = 1;

    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    
    // Allocate buffer for the entire image
    buffer = (unsigned char*)myalloc((*width) * (*height) * 3);
    if (!buffer) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return NULL;
    }

    // Allocate buffer for scanlines
    scanlines = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, 
         (*width) * 3, 1);

    // Read scanlines
    rowptr = buffer;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, scanlines, 1);
        memcpy(rowptr, scanlines[0], (*width) * 3);
        rowptr += (*width) * 3;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return buffer;
}

// Function to resize RGB buffer
unsigned char * resize_rgb_buffer(unsigned char* input_buffer, unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height){

    unsigned char * output_buffer = malloc(output_width * output_height * 3);
    if (!output_buffer) return NULL;

    // Simple nearest-neighbor scaling
    float x_ratio = input_width / (float)output_width;
    float y_ratio = input_height / (float)output_height;

    for (unsigned int y = 0; y < output_height; y++) {
        for (unsigned int x = 0; x < output_width; x++) {
            unsigned int px = (unsigned int)(x * x_ratio);
            unsigned int py = (unsigned int)(y * y_ratio);

            unsigned int output_pos = (y * output_width + x) * 3;
            unsigned int input_pos = (py * input_width + px) * 3;

            output_buffer[output_pos] = input_buffer[input_pos];        // R
            output_buffer[output_pos + 1] = input_buffer[input_pos + 1];// G
            output_buffer[output_pos + 2] = input_buffer[input_pos + 2];// B
        }
    }

    return output_buffer;
}

// Function to crop RGB buffer
unsigned char* crop_rgb_buffer(unsigned char* input_buffer,
                             unsigned int input_width,
                             unsigned int input_height,
                             unsigned int x,
                             unsigned int y,
                             unsigned int crop_width,
                             unsigned int crop_height) {
    // Boundary checking
    if (x + crop_width > input_width || y + crop_height > input_height)
        return NULL;

    unsigned char* output_buffer = malloc(crop_width * crop_height * 3);
    if (!output_buffer) return NULL;

    for (unsigned int cy = 0; cy < crop_height; cy++) {
        for (unsigned int cx = 0; cx < crop_width; cx++) {
            unsigned int output_pos = (cy * crop_width + cx) * 3;
            unsigned int input_pos = ((y + cy) * input_width + (x + cx)) * 3;

            output_buffer[output_pos] = input_buffer[input_pos];        // R
            output_buffer[output_pos + 1] = input_buffer[input_pos + 1];// G
            output_buffer[output_pos + 2] = input_buffer[input_pos + 2];// B
        }
    }

    return output_buffer;
}


// Helper macros used below 
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(x,min,max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
// Lanczos filter parameters
#define LANCZOS_RADIUS 3.0f

// Lanczos kernel function
static float lanczos_kernel(float x) {
    if (x == 0.0f) return 1.0f;
    if (x < -LANCZOS_RADIUS || x > LANCZOS_RADIUS) return 0.0f;
    
    x *= M_PI;
    float lanczos = (sinf(x) / x) * 
                    (sinf(x / LANCZOS_RADIUS) / (x / LANCZOS_RADIUS));
    return lanczos;
}

// High quality resize using Lanczos resampling
unsigned char* resize_rgb_lanczos(const unsigned char* input_buffer, unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height){

    unsigned char * output_buffer = malloc(output_width * output_height * 3);

    if (!output_buffer) return NULL;

    float x_ratio = (float)input_width / output_width;
    float y_ratio = (float)input_height / output_height;
    float x_scale = MIN(1.0f, x_ratio);
    float y_scale = MIN(1.0f, y_ratio);
    
    // For each output pixel
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            float center_x = (x + 0.5f) * x_ratio - 0.5f;
            float center_y = (y + 0.5f) * y_ratio - 0.5f;
            
            // Calculate sampling area
            int start_x = MAX(0, (int)(center_x - LANCZOS_RADIUS));
            int end_x = MIN(input_width - 1, (int)(center_x + LANCZOS_RADIUS));
            int start_y = MAX(0, (int)(center_y - LANCZOS_RADIUS));
            int end_y = MIN(input_height - 1, (int)(center_y + LANCZOS_RADIUS));
            
            float r = 0, g = 0, b = 0;
            float weight_sum = 0;
            
            // Accumulate weighted contributions
            for (int sy = start_y; sy <= end_y; sy++) {
                float dy = (sy - center_y) * y_scale;
                float y_weight = lanczos_kernel(dy);
                
                for (int sx = start_x; sx <= end_x; sx++) {
                    float dx = (sx - center_x) * x_scale;
                    float x_weight = lanczos_kernel(dx);
                    
                    float weight = x_weight * y_weight;
                    const unsigned char* pixel = 
                        &input_buffer[(sy * input_width + sx) * 3];
                    
                    r += pixel[0] * weight;
                    g += pixel[1] * weight;
                    b += pixel[2] * weight;
                    weight_sum += weight;
                }
            }
            
            // Normalize and write output pixel
            unsigned char* out_pixel = 
                &output_buffer[(y * output_width + x) * 3];
            if (weight_sum > 0) {
                out_pixel[0] = (unsigned char)CLAMP(r / weight_sum, 0, 255);
                out_pixel[1] = (unsigned char)CLAMP(g / weight_sum, 0, 255);
                out_pixel[2] = (unsigned char)CLAMP(b / weight_sum, 0, 255);
            }
        }
    }

    return output_buffer;
}


void displayPicture(Picture* pic, int timeout) {
    Display *display = XOpenDisplay(NULL);

	if(display == NULL){
		mylog(LOG_INFO,"");
		mylog(LOG_INFO,"     X11 Display not available.");
		mylog(LOG_INFO,"");
		sleep(2);
		return;
	}

    int screen = DefaultScreen(display);
    Window window = XCreateSimpleWindow(display, RootWindow(display, screen),
                                      10, 10, pic->width, pic->height, 1,
                                      BlackPixel(display, screen), WhitePixel(display, screen));

    // Set up handling for window close button
    Atom wmDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wmDeleteMessage, 1);

    // Set window states for foreground and active
    Atom wmState = XInternAtom(display, "_NET_WM_STATE", False);
    Atom wmStateAbove = XInternAtom(display, "_NET_WM_STATE_ABOVE", False);
    Atom wmStateFocused = XInternAtom(display, "_NET_WM_STATE_FOCUSED", False);


    // Create the image from our Picture structure
    XImage *ximage = XCreateImage(display, DefaultVisual(display, screen),
                                 DefaultDepth(display, screen), ZPixmap, 0,
                                 malloc(pic->width * pic->height * 4),
                                 pic->width, pic->height, 32, 0);

    // Copy pixel data from our Picture to XImage format
    for(int y = 0; y < pic->height; y++) {
        for(int x = 0; x < pic->width; x++) {
            Pixel p = pic->pic[y * pic->width + x];
            // Combine RGB into a single pixel value
            unsigned long pixel = (p.R << 16) | (p.G << 8) | p.B;
            XPutPixel(ximage, x, y, pixel);
        }
    }

    // Listen for both exposure events (for redrawing) and client messages (for window close)
    XSelectInput(display, window, ExposureMask | KeyPressMask | StructureNotifyMask);
    XMapWindow(display, window);

    
    // Raise and focus window
    XRaiseWindow(display, window);
    XSetInputFocus(display, window, RevertToParent, CurrentTime);


    GC gc = XCreateGC(display, window, 0, NULL);

    // Flush to ensure changes are applied
    XFlush(display);



    while(timeout!=0) {
        XEvent e;
        XNextEvent(display, &e);
        
        if(e.type == Expose) {
            // Redraw the image when window needs updating
            XPutImage(display, window, gc, ximage, 0, 0, 0, 0, pic->width, pic->height);
        }
        else if(e.type == ClientMessage) {
            // Check if it's a window close message
            if((Atom)e.xclient.data.l[0] == wmDeleteMessage) {
                break;  // Exit the loop if window close button was clicked
            }
        }
        else if(e.type == KeyPress) {
            break;  // Exit on any key press (keeping this feature from original code)
        }
	timeout--;
	sleep(1);
    }

    // Clean up resources
    XDestroyImage(ximage);  // This also frees the malloc'd data
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
}





void displayPicture_resize(Picture * pic, int width, int height, int timeout){
	Picture * newpic = malloc(sizeof(Picture));
	if((width >= pic->width)  ||  (height >= pic->height)){
		newpic->pic = (Pixel *)resize_rgb_buffer((unsigned char *)pic->pic, pic->width, pic->height, width, height);	//looks better!
	}
	else{
		newpic->pic = (Pixel *)resize_rgb_lanczos((unsigned char *)pic->pic, pic->width, pic->height, width, height);
	}
	newpic->width = width;
	newpic->height = height;
	displayPicture(newpic, timeout);
	free(newpic->pic);
	free(newpic);
}


void draw_rectangle(Picture* picture, int y_min, int x_min, int y_max, int x_max, byte R, byte G, byte B, int thickness, float fill_alpha) {
    // Input validation
    if (!picture || !picture->pic || thickness < 0 || 
        fill_alpha < 0.0f || fill_alpha > 1.0f) return;
    
    // Ensure coordinates are within bounds
    y_min = (y_min < 0) ? 0 : (y_min >= picture->height) ? picture->height - 1 : y_min;
    x_min = (x_min < 0) ? 0 : (x_min >= picture->width) ? picture->width - 1 : x_min;
    y_max = (y_max < 0) ? 0 : (y_max >= picture->height) ? picture->height - 1 : y_max;
    x_max = (x_max < 0) ? 0 : (x_max >= picture->width) ? picture->width - 1 : x_max;

    // Ensure min is less than max
    if (y_min > y_max) { int temp = y_min; y_min = y_max; y_max = temp; }
    if (x_min > x_max) { int temp = x_min; x_min = x_max; x_max = temp; }

    // Adjust thickness if it would exceed rectangle dimensions
    int rect_height = y_max - y_min + 1;
    int rect_width = x_max - x_min + 1;
    thickness = (thickness > rect_height/2) ? rect_height/2 : thickness;
    thickness = (thickness > rect_width/2) ? rect_width/2 : thickness;

    // Draw the borders (with full color)
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            // Check if current pixel is within border thickness
            if (y < y_min + thickness || y > y_max - thickness ||
                x < x_min + thickness || x > x_max - thickness) {
                int idx = y * picture->width + x;
                picture->pic[idx].R = R;
                picture->pic[idx].G = G;
                picture->pic[idx].B = B;
            }
        }
    }

    // Fill the interior with transparency if fill_alpha > 0
    if (fill_alpha > 0.0f) {
        for (int y = y_min + thickness; y <= y_max - thickness; y++) {
            for (int x = x_min + thickness; x <= x_max - thickness; x++) {
                int idx = y * picture->width + x;
                
                // Linear interpolation for transparency
                picture->pic[idx].R = (byte)(picture->pic[idx].R * (1.0f - fill_alpha) + R * fill_alpha);
                picture->pic[idx].G = (byte)(picture->pic[idx].G * (1.0f - fill_alpha) + G * fill_alpha);
                picture->pic[idx].B = (byte)(picture->pic[idx].B * (1.0f - fill_alpha) + B * fill_alpha);
            }
        }
    }
}

