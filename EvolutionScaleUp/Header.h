#pragma once
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>
#include <random>
#include <algorithm>

void PrintMatrixF16(__half* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", __half2float(arr[i * cols + j]));
		printf("\n");
	}
	printf("\n");
}

__global__ void CurandNormalizeF16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * 0.0000152590218967f * range + min);
}

void CurandGenerateUniformF16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizeF16 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, max - min);
}

namespace GLOBAL
{
	const float ONEF32 = 1.0f;
	const float ZEROF32 = 0.0f;

	const __half ONEF16 = __float2half(1.0f);
	const __half ZEROF16 = __float2half(0.0f);

	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;
	curandGenerator_t curandGenerator;
	cudnnActivationDescriptor_t reluActivationDescriptor;

	std::random_device randomDevice;
	std::mt19937 randomEngine(randomDevice());
	std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);

	const uint32_t ACTIONS = 3;
	const uint32_t ACTION_BYTES = ACTIONS << 1;
	const uint32_t SCORES = ACTIONS * ACTIONS;
	const uint32_t GENERATIONS = 1000;
	
	const uint32_t AGENTS = 256;
	const uint32_t AGENTS_PER_GAME = 2;
	const uint32_t BATCHES = 10;
	const uint32_t GAMES = AGENTS * BATCHES / AGENTS_PER_GAME;
	const float TOP_PERCENT = 0.1f;
	const uint32_t TOP_AGENTS = AGENTS * TOP_PERCENT;
	
	// tic tac toe
	const float scores[SCORES] =
	{
		0.0f, -1.0f, 1.0f,
		1.0f, 0.0f, -1.0f,
		-1.0f, 1.0f, 0.0f
	};

	__half* cpuSoftmaxTensor;
	__half* gpuBiasMutationTensor;

	void Init()
	{
		cublasCreate(&cublasHandle);
		cudnnCreate(&cudnnHandle);
		curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(curandGenerator, randomEngine());
		cudnnCreateActivationDescriptor(&reluActivationDescriptor);
		cudnnSetActivationDescriptor(reluActivationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
		
		cpuSoftmaxTensor = new __half[ACTIONS];
		cudaMalloc(&gpuBiasMutationTensor, ACTION_BYTES);
	}

	void Destroy()
	{
		cublasDestroy(cublasHandle);
		cudnnDestroy(cudnnHandle);
		curandDestroyGenerator(curandGenerator);
		cudnnDestroyActivationDescriptor(reluActivationDescriptor);

		delete[] cpuSoftmaxTensor;
		cudaFree(gpuBiasMutationTensor);
	}
}