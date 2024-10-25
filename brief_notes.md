# Paligemma Notes

## From CLIP to SigLIP

Problems with CLIP: Cross entropy loss 
1. into infinity, not numerically stable for calculating exponential results of distribution
2. computational expensive

how to make cross-entropy loss numerically stable?

$$
Softmax = \frac{c\cdot e^{a_i}}{c\cdot \sum_{k=1}^N e^{a_k}} = \frac{e^{a_i + \log c}}{\sum_{k=1} e^{a_k + \log c}}\\
make \log c = - \max_{i} a_i
$$

Code implementation of CLIP:

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed 
# W_t[d_t, d_e] - learned proj of text to embed 
# t             - learned temperature parameter 

# extract feature representations of each modality
I_f = image_encoder(I)   # [n, d_i] 
T_f = text_encoder(T)   # [n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) 
# scaled pairwise cosine similaries [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric cross entropy loss function
labels = np.arrange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t) / 2
```


SigLIP:

use sigmoid function to replace softmax loss;

Code implementation of SigLIP:




## Vision Transformer 

