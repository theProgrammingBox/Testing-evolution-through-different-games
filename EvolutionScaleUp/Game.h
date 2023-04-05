#pragma once
#include "Agent.cuh"

class Game
{
public:
	Game();
	~Game();

	void Play(Agent* agent1, Agent* agent2);
};

Game::Game()
{
}

Game::~Game()
{
}

void Game::Play(Agent* agent1, Agent* agent2)
{
	uint32_t action1 = agent1->Forward();
	uint32_t action2 = agent2->Forward();
	//printf("Agent 1: %d, Agent 2: %d\n", action1, action2);
	agent1->score += GLOBAL::scores[action1 * GLOBAL::ACTIONS + action2];
	agent2->score += GLOBAL::scores[action2 * GLOBAL::ACTIONS + action1];
	//printf("idx1: %d, idx2: %d\n", action1 * GLOBAL::ACTIONS + action2, action2 * GLOBAL::ACTIONS + action1);
	//printf("Score1: %f, Score2: %f\n", GLOBAL::scores[action1 * GLOBAL::ACTIONS + action2], GLOBAL::scores[action2 * GLOBAL::ACTIONS + action1]);
}