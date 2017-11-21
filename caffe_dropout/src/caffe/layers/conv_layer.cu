#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ConvForward(const int n, int row_num, int row_dim, 
    const unsigned int* mask, const unsigned int threshold, 
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = out[index] * (mask[(index/row_dim) % row_num] > threshold);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  
  // begin init mask.
  int conv_out_channels = BaseConvolutionLayer<Dtype>::get_conv_out_channels();
  int conv_out_spatial_dim = BaseConvolutionLayer<Dtype>::get_conv_out_spatial_dim();
  unsigned int *mask = static_cast<unsigned int*>(BaseConvolutionLayer<Dtype>::get_mask_gpu());
  int threshold = 0.5; // 80% are 1.
  unsigned int uint_thres = static_cast<unsigned int>(UINT_MAX * threshold);
  caffe_gpu_rng_uniform(conv_out_channels, mask);
  // end.
  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }    
    }
    
    // set 20% rows as 0.
    const int count = top[i]->count();
 	ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    	count, conv_out_channels, conv_out_spatial_dim, mask, uint_thres, top_data);   
  }
}

template <typename Dtype>
__global__ void ConvBackward(const int n, int row_num, int row_dim, 
    const unsigned int* mask, const unsigned int threshold, 
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = out[index] * (mask[(index/row_dim) % row_num] > threshold);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  
   // begin init mask.
  int conv_out_channels = BaseConvolutionLayer<Dtype>::get_conv_out_channels();
  int conv_out_spatial_dim = BaseConvolutionLayer<Dtype>::get_conv_out_spatial_dim();
  unsigned int *mask = static_cast<unsigned int*>(BaseConvolutionLayer<Dtype>::get_mask_gpu());
  int threshold = 0.5; // 80% are 1.
  unsigned int uint_thres = static_cast<unsigned int>(UINT_MAX * threshold);
  // end.
  
  
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_diff = top[i]->mutable_gpu_diff();
    
    // set 20% rows as 0.
    const int count = top[i]->count();
 	ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    	count, conv_out_channels, conv_out_spatial_dim, mask, uint_thres, top_diff);  
       
    
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }   
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
