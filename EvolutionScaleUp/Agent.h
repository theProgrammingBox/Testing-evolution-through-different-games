#pragma once
#include "Header.cuh"

class Agent
{
public:
	__half* gpuBiasTensor;
	__half* gpuSoftmaxTensor;
	cudnnTensorDescriptor_t biasTensorDescriptor;
	float score;
	
	Agent();
	~Agent();
	uint32_t Forward() const;
	void Mutate(Agent* parentAgent);
	void PrintDistribution() const;
};

Agent::Agent()
{
	cudaMalloc(&gpuBiasTensor, GLOBAL::ACTION_BYTES);
	cudaMalloc(&gpuSoftmaxTensor, GLOBAL::ACTION_BYTES);
	
	CurandGenerateUniformF16(GLOBAL::curandGenerator, gpuBiasTensor, GLOBAL::ACTIONS, -0.1f, 0.1f);
	
	cudnnCreateTensorDescriptor(&biasTensorDescriptor);
	cudnnSetTensor4dDescriptor(biasTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, GLOBAL::ACTIONS);

	score = 0.0f;
}

Agent::~Agent()
{
	cudaFree(gpuBiasTensor);
	cudaFree(gpuSoftmaxTensor);
	cudnnDestroyTensorDescriptor(biasTensorDescriptor);
}

uint32_t Agent::Forward() const
{
	cudnnSoftmaxForward
	(
		GLOBAL::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
		&GLOBAL::ONEF32, biasTensorDescriptor, gpuBiasTensor,
		&GLOBAL::ZEROF32, biasTensorDescriptor, gpuSoftmaxTensor
	);

	cudaMemcpy(GLOBAL::cpuSoftmaxTensor, gpuSoftmaxTensor, GLOBAL::ACTION_BYTES, cudaMemcpyDeviceToHost);
	
	float probability = GLOBAL::randomFloat(GLOBAL::randomEngine);
	for (uint32_t i = GLOBAL::ACTIONS; i--;)
	{
		probability -= __half2float(GLOBAL::cpuSoftmaxTensor[i]);
		if (probability <= 0.0f)
			return i;
	}
	return 0;
}

void Agent::Mutate(Agent* parentAgent)
{
	cudaMemcpy(gpuBiasTensor, parentAgent->gpuBiasTensor, GLOBAL::ACTION_BYTES, cudaMemcpyDeviceToDevice);

	CurandGenerateUniformF16(GLOBAL::curandGenerator, GLOBAL::gpuBiasMutationTensor, GLOBAL::ACTIONS, -0.01f, 0.01f);

	cublasAxpyEx
	(
		GLOBAL::cublasHandle, GLOBAL::ACTIONS,
		&GLOBAL::ONEF32, CUDA_R_32F,
		GLOBAL::gpuBiasMutationTensor, CUDA_R_16F, 1,
		gpuBiasTensor, CUDA_R_16F, 1,
		CUDA_R_32F
	);
}

void Agent::PrintDistribution() const
{
	cudaMemcpy(GLOBAL::cpuSoftmaxTensor, gpuSoftmaxTensor, GLOBAL::ACTION_BYTES, cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i < GLOBAL::ACTIONS; i++)
		printf("%f ", __half2float(GLOBAL::cpuSoftmaxTensor[i]));
	printf("\n");
}