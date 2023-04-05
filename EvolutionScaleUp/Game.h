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
	agent1->score += GLOBAL::scores[action1 * GLOBAL::AGENTS_PER_GAME + action2];
	agent2->score += GLOBAL::scores[action2 * GLOBAL::AGENTS_PER_GAME + action1];
}