/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

// 定义一个结构体，用于存储Transformer模型的配置参数
typedef struct {
    int dim; // transformer的维度
    int hidden_dim; // 用于前馈网络层的隐藏维度
    int n_layers; // 层数
    int n_heads; // 查询头的数量
    int n_kv_heads; // 键/值头的数量（可以小于查询头，因为可能有多查询）
    int vocab_size; // 词汇表大小，通常为256（字节级）
    int seq_len; // 最大序列长度
} Config;

// 定义一个结构体，用于存储Transformer模型的权重
typedef struct {
    // 词嵌入表
    float* token_embedding_table;    // (词汇表大小, 维度)
    // rmsnorm的权重
    float* rms_att_weight; // (层, 维度) rmsnorm权重
    float* rms_ffn_weight; // (层, 维度) rmsnorm权重
    // matmul的权重。注意维度 == n_heads * head_size
    float* wq; // (层, 维度, n_heads * head_size)
    float* wk; // (层, 维度, n_kv_heads * head_size)
    float* wv; // (层, 维度, n_kv_heads * head_size)
    float* wo; // (层, n_heads * head_size, 维度)
    // 前馈网络的权重
    float* w1; // (层, hidden_dim, 维度)
    float* w2; // (层, 维度, hidden_dim)
    float* w3; // (层, hidden_dim, 维度)
    // 最终的rmsnorm
    float* rms_final_weight; // (维度,)
    // (可选) 最后一层的分类器权重，用于logits
    float* wcls;
} TransformerWeights;

// 定义一个结构体，用于存储Transformer模型的运行状态
typedef struct {
    // 当前波次的激活值
    float *x; // 当前时间戳的激活值 (维度,)
    float *xb; // 残差分支内的激活值 (维度,)
    float *xb2; // 额外的缓冲区，仅为方便 (维度,)
    float *hb; // 前馈网络中隐藏维度的缓冲区 (hidden_dim,)
    float *hb2; // 前馈网络中隐藏维度的缓冲区 (hidden_dim,)
    float *q; // 查询 (维度,)
    float *k; // 键 (维度,)
    float *v; // 值 (维度,)
    float *att; // 用于得分/注意力值的缓冲区 (n_heads, seq_len)
    float *logits; // 输出logits
    // kv缓存
    float* key_cache;   // (层, seq_len, 维度)
    float* value_cache; // (层, seq_len, 维度)
} RunState;

// 定义一个结构体，用于存储Transformer模型的完整信息
typedef struct {
    Config config; // 模型的超参数（蓝图）
    TransformerWeights weights; // 模型的权重
    RunState state; // 正向传播中“波”的激活值的缓冲区
    // 一些额外的状态，用于正确清理内存映射（唉）
    int fd; // 内存映射的文件描述符
    float* data; // 内存映射的数据指针
    ssize_t file_size; // 检查点文件的大小（字节）
} Transformer;

// 为RunState结构体分配内存
void malloc_run_state(RunState* s, Config* p) {
    // 使用calloc而不是malloc，以确保内存初始化为0，使内存检测工具valgrind满意
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; // 计算键/值维度
    s->x = calloc(p->dim, sizeof(float)); // 为激活值x分配内存
    s->xb = calloc(p->dim, sizeof(float)); // 为残差分支激活值xb分配内存
    s->xb2 = calloc(p->dim, sizeof(float)); // 为额外缓冲区xb2分配内存
    s->hb = calloc(p->hidden_dim, sizeof(float)); // 为前馈网络隐藏维度hb分配内存
    s->hb2 = calloc(p->hidden_dim, sizeof(float)); // 为前馈网络隐藏维度hb2分配内存
    s->q = calloc(p->dim, sizeof(float)); // 为查询q分配内存
    s->k = calloc(p->dim, sizeof(float)); // 为键k分配内存
    s->v = calloc(p->dim, sizeof(float)); // 为值v分配内存
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float)); // 为注意力值att分配内存
    s->logits = calloc(p->vocab_size, sizeof(float)); // 为logits分配内存
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)); // 为键缓存分配内存
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)); // 为值缓存分配内存

    // 确保所有malloc调用都成功
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n"); // 打印错误信息
        exit(EXIT_FAILURE); // 退出程序
    }
}

// 释放RunState结构体的内存
void free_run_state(RunState* s) {
    free(s->x); // 释放激活值x的内存
    free(s->xb); // 释放残差分支激活值xb的内存
    free(s->xb2); // 释放额外缓冲区xb2的内存
    free(s->hb); // 释放前馈网络隐藏维度hb的内存
    free(s->hb2); // 释放前馈网络隐藏维度hb2的内存
    free(s->q); // 释放查询q的内存
    free(s->k); // 释放键k的内存
    free(s->v); // 释放值v的内存
    free(s->att); // 释放注意力值att的内存
    free(s->logits); // 释放logits的内存
    free(s->key_cache); // 释放键缓存的内存
    free(s->value_cache); // 释放值缓存的内存
}

// 将权重映射到内存
void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads; // 计算头大小
    // 确保下面的乘法使用64位整数以适应13B+模型的参数计数
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr; // 词嵌入表指向ptr
    ptr += p->vocab_size * p->dim; // 更新指针位置
    w->rms_att_weight = ptr; // rms_att_weight指向ptr
    ptr += n_layers * p->dim; // 更新指针位置
    w->wq = ptr; // wq指向ptr
    ptr += n_layers * p->dim * (p->n_heads * head_size); // 更新指针位置
    w->wk = ptr; // wk指向ptr
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size); // 更新指针位置
    w->wv = ptr; // wv指向ptr
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size); // 更新指针位置
    w->wo = ptr; // wo指向ptr
    ptr += n_layers * (p->n_heads * head_size) * p->dim; // 更新指针位置
    w->rms_ffn_weight = ptr; // rms_ffn_weight指向ptr
    ptr += n_layers * p->dim; // 更新指针位置
    w->w1 = ptr; // w1指向ptr
    ptr += n_layers * p->dim * p->hidden_dim; // 更新指针位置
    w->w2 = ptr; // w2指向ptr
    ptr += n_layers * p->hidden_dim * p->dim; // 更新指针位置
    w->w3 = ptr; // w3指向ptr
    ptr += n_layers * p->dim * p->hidden_dim; // 更新指针位置
    w->rms_final_weight = ptr; // rms_final_weight指向ptr
    ptr += p->dim; // 更新指针位置

    // 跳过RoPE模型中使用的频谱特征的内存位置
    ptr += p->seq_len * head_size / 2; // 跳过freq_cis_real
    ptr += p->seq_len * head_size / 2; // 跳过freq_cis_imag

    // 如果共享权重，则wcls指向词嵌入表，否则指向新的内存位置
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// 从检查点文件读取模型的配置和权重
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb"); // 以二进制读取模式打开文件
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", checkpoint); 
        exit(EXIT_FAILURE); 
    }
    // 读取配置头部
    if (fread(config, sizeof(Config), 1, file) != 1) { 
        exit(EXIT_FAILURE); 
    }
    // 负的词汇表大小是一个hack，用于表示权重不共享
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size); // 确保词汇表大小为正数
    // 确定文件大小
    fseek(file, 0, SEEK_END); // 移动文件指针到文件末尾
    *file_size = ftell(file); // 获取文件大小（字节）
    fclose(file); // 关闭文件
    // 将Transformer权重内存映射到数据指针
    *fd = open(checkpoint, O_RDONLY); // 以只读模式打开
    if (*fd == -1) { 
        fprintf(stderr, "open failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { 
        fprintf(stderr, "mmap failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float); // 计算权重数据的起始位置
    memory_map_weights(weights, config, weights_ptr, shared_weights); // 映射权重
}

// 构建Transformer模型
void build_transformer(Transformer *t, char* checkpoint_path) {
    // 从检查点读取Config和Weights
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // 为RunState缓冲区分配内存
    malloc_run_state(&t->state, &t->config);
}

// 释放Transformer模型
void free_transformer(Transformer* t) {
    // 关闭内存映射
    if (t->data != MAP_FAILED) { 
        munmap(t->data, t->file_size); 
    }
    if (t->fd != -1) { 
        close(t->fd); 
    }
    // 释放RunState缓冲区的内存
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// 神经网络块；Transformer的动态

// RMSNorm层的实现
void rmsnorm(float* o, float* x, float* weight, int size) {
    // 计算平方和
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size; // 计算平均值
    ss += 1e-5f; // 添加小常数以防止除以零
    ss = 1.0f / sqrtf(ss); // 计算平方根的倒数
    // 归一化并缩放
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]); // 输出 = 权重 * 归一化输入
    }
}

// Softmax层的实现
void softmax(float* x, int size) {
    // 寻找最大值（为了数值稳定性）
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // 计算指数和求和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val); // 指数计算
        sum += x[i]; // 求和
    }
    // 归一化
    for (int i = 0; i < size; i++) {
        x[i] /= sum; // 输出 = 输入 / 总和
    }
}

// 矩阵乘法的实现
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // 这是花费时间最多的函数
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j]; // 计算矩阵乘法
        }
        xout[i] = val; // 将结果存储在输出数组中
    }
}

// 前向传播函数
float* forward(Transformer* transformer, int token, int pos) {

    // 一些方便的变量
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // 键/值共享的整数倍数
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // 将token嵌入复制到x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // 前向传播所有层
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // 注意力RMSNorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // 键和值指向kv缓存
        int loff = l * p->seq_len * kv_dim; // kv缓存层偏移量
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv矩阵乘法
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE相对位置编码：在每个头中复数旋转q和k
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // 旋转的向量数？2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // 要旋转的向量（查询或键）
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // 多头注意力。遍历所有头
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // 获取此头的查询向量
            float* q = s->q + h * head_size;
            // 注意力分数为此头
            float* att = s->att + h * p->seq_len;
            // 遍历所有时间步，包括当前时间步
            for (int t = 0; t <= pos; t++) {
                // 获取此头和此时间步的键向量
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // 计算注意力分数作为q和k的点积
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // 将分数保存到注意力缓冲区
                att[t] = score;
            }

            // 将分数进行softmax以获得注意力权重，从0到pos（包括）
            softmax(att, pos + 1);

            // 加权和的值，存储回xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // 获取此头和此时间步的值向量
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // 获取此时间步的注意力权重
                float a = att[t];
                // 将加权值累积到xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // 最终矩阵乘法以获得注意力的输出
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // 残差连接回x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // 现在对于PyTorch中的FFN，我们有：self.w2(F.silu(self.w1(x)) * self.w3(x))
        // 首先计算self.w1(x)和self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU非线性激活函数
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), 其中σ(x)是逻辑sigmoid函数
            val *= (1.0f / (1.0f + expf(-val)));
            // 与w3(x)逐元素相乘
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // 最终矩阵乘法以获得ffn的输出
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // 残差连接
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // 最终RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // 分类器到logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------

// 字节对编码（BPE）分词器，用于字符串和token之间的转换

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // 存储所有单字节字符串
} Tokenizer;

// 比较函数，用于qsort
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

// 构建分词器
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // 应该在分词器文件中写入vocab_size...唉
    t->vocab_size = vocab_size;
    // malloc空间以保存分数和字符串
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // 延迟初始化
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // 读取文件
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { 
        fprintf(stderr, "couldn't load %s\n", tokenizer_path); 
        exit(EXIT_FAILURE); 
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "failed read\n"); 
        exit(EXIT_FAILURE); 
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE); 
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE); 
        }
        t->vocab[i][len] = '\0'; // 添加字符串终止符
    }
    fclose(file);
}

// 释放分词器
void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { 
        free(t->vocab[i]); 
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

// 解码
char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // 跟随BOS（1）token，sentencepiece解码器会剥离任何前导空格（见PR #89）
    if (prev_token == 1 && piece[0] == ' ') { 
        piece++; 
    }
    // 注意，一些token指定原始字节，看起来像e.g. '<0x01>'
    // 解析此并转换并返回实际的字节
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

// 安全打印
void safe_printf(char *piece) {
    // piece可能是原始字节token，我们只想打印可打印字符或空格
    // 因为其他一些字节可能是各种控制代码，退格等
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // 坏字节，不要打印它
        }
    }
    printf("%s", piece);
}

// 字符串查找
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // 在vocab中高效地找到str的完美匹配，返回其索引或-1如果没有找到
    TokenIndex tok = { .str = str }; // 作为要搜索的键
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// 将字符串编码为token数组
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // 将字符串text（输入）编码到预分配的上限tokens[]数组中
    // 如果bos!=0，则在前面添加BOS token（=1），如果eos!=0，则在后面添加EOS token（=2）
    if (text == NULL) { 
        fprintf(stderr, "cannot encode NULL text\n"); 
        exit(EXIT_FAILURE); 
    }

    if (t->sorted_vocab == NULL) {
        // 延迟初始化，分配并排序词汇表
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // 创建一个临时缓冲区，将存储始终两个连续token的合并候选项
    // *2用于连接，+1用于空终止符+2用于UTF8（如果max_token_length为1）
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // 从0个token开始
    *n_tokens = 0;

    // 添加可选的BOS（=1）token，如果需要
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix默认为true
    // 因此，如果text!=""，则在输入字符串前添加一个虚拟前缀token
    // TODO: 我敢肯定这在一般情况下是不正确的，但我没有精力去阅读更多的sentencepiece代码来弄清楚它在做什么
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // 好的UTF-8时间。这将变得混乱。这里是维基百科的参考：
    // 代码点 ↔ UTF-8转换
    // 第一个代码点	最后一个代码点	字节1	字节2	字节3	字节4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // 处理输入字符串的原始（UTF-8）字节序列
    for (char *c = text; *c != '\0'; c++) {

        // 如果当前字节是ASCII或前导字节，则重置缓冲区
        // 0xC0是11000000，所以(*c & 0xC0)保留前两位并将其余位清零
        // 0x80是10000000
        // 在UTF-8中，所有后续字节都以"10"开头的前两位
        // 所以在英语中这是："如果这个字节不是后续字节"
        if ((*c & 0xC0) != 0x80) {
            // 这个字节必须是前导字节（11...）或ASCII字符（0x...）
            // => 重置我们的位置，因为我们正在开始一个新的UTF-8代码点
            str_len = 0;
        }

        // 将当前字节追加到缓冲区
        str_buffer[str_len++] = *c; // ++是后增量，在这行之后增加
        str_buffer[str_len] = '\0';

        // 虽然下一个字符是后续字节，但继续追加
        // 但如果它们太多，只停止以避免超出str_buffer大小。
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // 好的c+1不是后续字节，所以我们已经读取了一个完整的代码点
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // 我们在词汇表中找到了这个代码点，将其作为token添加
            tokens[(*n_tokens)++] = id;
        } else {
            // 字节回退编码：将每个字节编码为一个token
            // +3是因为前3个词汇元素是<unk>，<s>，</s>
            // 所以个别字节只从索引3开始
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // 防止一系列杂乱的UTF8后续字节
    }

    // 每次迭代合并最佳连续对，根据vocab_scores中的分数
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // 检查我们是否可以合并对（tokens[i]，tokens[i+1]）
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // 这个合并对存在于词汇表中！记录其分数和位置
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // 我们找不到更多的对来合并，所以我们完成了
        }

        // 合并连续对（best_idx，best_idx+1）到新token best_id
        tokens[best_idx] = best_id;
        // 删除位置best_idx+1的token，将整个序列向后移动1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token长度减少
    }

    // 添加可选的EOS（=2）token，如果需要
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------

// 采样器，它接收logits并返回一个采样的token
// 采样可以通过几种方式完成：贪婪的argmax、采样、top-p采样

typedef struct {
    float prob;
    int index;
} ProbIndex; // 在top-p采样期间用于排序概率的结构体

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // 在top-p采样中使用的缓冲区
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// 返回具有最高概率的索引
int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// 从概率中采样索引（它们必须总和为1！）
// coin是[0, 1)中的一个随机数，通常来自random_f32()
int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // 以防万一四舍五入错误
}

// 比较函数，用于qsort
int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

// top-p采样（或“核采样”）从超过概率topp的最小token集中采样
// 这样我们永远不采样概率非常低的token，不太可能“偏离轨道”
// coin是[0, 1)中的一个随机数，通常来自random_f32()
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    // 快速排序索引，按概率降序
    // 小于(1 - topp) / (n - 1)的值不能是结果的一部分
    // 因此在排序之前将这些作为候选项裁剪掉以提高效率
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // 截断累积概率超过topp的列表
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // 以防万一四舍五入错误考虑所有元素
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // 通过包含last_idx，我们已经超过了topp
        }
    }

    // 从截断的列表中采样
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // 以防万一四舍五入错误
}

// 构建采样器
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // 缓冲区仅在核采样时使用；可能不需要，但它很小
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

// 释放采样器
void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A 
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random_f32()生成的随机float32在[0,1)中
float random_f32(unsigned long long *state) { 
    return (random_u32(state) >> 8) / 16777216.0f;
}

// 给定logits和一些超参数，采样token
int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        // 贪婪argmax采样：选择概率最高的token
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // 将温度应用于logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // 将logits应用softmax以获得下一个token的概率
        softmax(logits, sampler->vocab_size);
        // 抛一枚（float）硬币（这是我们采样的熵源）
        float coin = random_f32(&sampler->rng_state);
        // 我们从这个分布中采样以获得下一个token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // 简单地从预测的概率分布中采样
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p（核）采样，将最不可能的tokens限制为零
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// 工具：时间

// 返回以毫秒为单位的时间，用于基准测试模型速度
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time); // 获取当前时间
    return time.tv_sec * 1000 + time.tv_nsec / 1000000; // 将时间转换为毫秒
}

// ----------------------------------------------------------------------------
// generation loop

// 生成循环

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // 将（字符串）提示编码为token序列
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // 开始主循环
    long start = 0;  // 用于计时我们的代码，仅在第一次迭代后初始化
    int next;        // 将存储序列中的下一个token
    int token = prompt_tokens[0]; // 用提示中的第一个token启动
    int pos = 0;     // 序列中的位置
    while (pos < steps) {

        // 前向传播transformer以获取下一个token的logits
        float* logits = forward(transformer, token, pos);

        // 推进状态机
        if (pos < num_prompt_tokens - 1) {
            // 如果我们仍在处理输入提示，强制执行下一个提示token
            next = prompt_tokens[pos + 1];
        } else {
            // 否则从logits中采样下一个token
            next = sample(sampler, logits);
        }
        pos++;

        // 数据依赖的终止条件：BOS (=1) token限定序列
        if (next == 1) { break; }

        // 打印token作为字符串，使用Tokenizer对象解码它
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // 与printf("%s", piece)相同，但跳过“不安全”的字节
        fflush(stdout);
        token = next;

        // 在这里初始化计时器，因为第一次迭代可能会更慢
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // 报告实现的tok/s（pos-1因为计时器在第一次迭代后开始）
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // 从stdin读取一行，不包括\n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // 去掉换行符
        }
    }
}

// ----------------------------------------------------------------------------
// 聊天循环
// 我手动检查了一些与python参考相比的聊天对话的tokens，看起来没问题，
// 但这个没有经过彻底测试，并且没有安全实现，目前更多的是一个概念验证。

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // 用于从stdin读取系统提示和用户提示的缓冲区
    // 你会注意到它们目前设置得有些随意和不安全
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // 开始主循环
    int8_t user_turn = 1; // 用户开始
    int next;        // 将存储序列中的下一个token
    int token;       // 存储要输入transformer的当前token
    int prev_token;
    int pos = 0;     // 序列中的位置
    while (pos < steps) {

        // 当轮到用户为对话贡献tokens时...
        if (user_turn) {
            // 获取位置0的（可选的）系统提示
            if (pos == 0) {
                // 在位置0，用户也可以贡献系统提示
                if (cli_system_prompt == NULL) {
                    // 系统提示没有传入，尝试从stdin获取
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // 系统提示已传入，使用它
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // 获取用户提示
            if (pos == 0 && cli_user_prompt != NULL) {
                // 用户提示对于位置0已传入，使用它
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // 否则从stdin获取用户提示
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // 将用户/系统提示渲染成Llama 2 Chat模式
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // 将渲染后的提示编码为tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // 重置用户索引
            user_turn = 0;
            printf("Assistant: ");
        }

        // 确定下一个要输入transformer的token
        if (user_idx < num_prompt_tokens) {
            // 如果我们仍在处理输入提示，强制执行下一个提示token
            token = prompt_tokens[user_idx++];
        } else {
            // 否则使用上一轮采样的下一个token
            token = next;
        }
        // EOS (=2) token结束助手回合
        if (token == 2) { user_turn = 1; }

        // 前向传播transformer以获取下一个token的logits
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // 助手正在回应，因此打印其输出
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // 与printf("%s", piece)相同，但跳过“不安全”的字节
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI，如果不需要测试则包含

#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // 默认参数
    char *checkpoint_path = NULL;  // 例如 out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = 贪婪确定性。1.0 = 原始。不要设置更高
    float topp = 0.9f;          // 核采样中的top-p。1.0 = 关闭。0.9 效果不错，但速度较慢
    int steps = 256;            // 运行的步数
    char *prompt = NULL;        // 提示字符串
    unsigned long long rng_seed = 0; // 默认用时间种子rng
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // 聊天模式中的（可选的）系统提示

    // 简单的C参数解析，这样我们就可以覆盖上面的默认值
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // 做一些基本验证
        if (i + 1 >= argc) { error_usage(); } // 标志后必须有参数
        if (argv[i][0] != '-') { error_usage(); } // 必须以短划线开始
        if (strlen(argv[i]) != 2) { error_usage(); } // 必须是 -x（一个短划线，一个字母）
        // 读取参数
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // 参数验证/覆盖
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // 通过模型 .bin 文件构建 Transformer
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // 重写为 ~最大长度

    // 通过分词器 .bin 文件构建 Tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 构建 Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // 运行!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // 内存和文件句柄清理
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
