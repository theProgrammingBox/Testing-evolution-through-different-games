#include "Game.cuh"

int main()
{
	GLOBAL::Init();
	printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
	printf("cuBLAS version: %d.%d.%d\n", CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH);
	printf("cuRAND version: %d.%d.%d\n", CURAND_VERSION / 1000, (CURAND_VERSION % 1000) / 100, CURAND_VERSION % 100);
	printf("\n");

	Agent* agents[GLOBAL::AGENTS];
	Game games[GLOBAL::GAMES];

	for (uint32_t i = GLOBAL::AGENTS; i--;)
		agents[i] = new Agent();

	for (uint32_t generation = 0; generation < GLOBAL::GENERATIONS; generation++)
	{
		for (uint32_t i = GLOBAL::GAMES; i--;)
		{
			uint32_t agent1 = GLOBAL::randomEngine() % GLOBAL::AGENTS;
			uint32_t agent2 = GLOBAL::randomEngine() % GLOBAL::AGENTS;
			games[i].Play(agents[agent1], agents[agent2]);
		}

		// sort agents by score
		std::sort(agents, agents + GLOBAL::AGENTS, [](Agent* a, Agent* b) { return a->score > b->score; });

		for (uint32_t i = GLOBAL::TOP_AGENTS; i < GLOBAL::AGENTS; i++)
		{
			uint32_t parent = GLOBAL::randomEngine() % GLOBAL::TOP_AGENTS;
			agents[i]->Mutate(agents[parent]);
		}
		
		float averageScore = 0.0f;
		for (uint32_t i = 0; i < GLOBAL::AGENTS; i++)
		{
			averageScore += agents[i]->score;
			agents[i]->score = 0.0f;
		}
		averageScore /= GLOBAL::AGENTS;
		printf("Generation %d: average score = %f\n", generation, averageScore);
	}

	GLOBAL::Destroy();
	return 0;
}