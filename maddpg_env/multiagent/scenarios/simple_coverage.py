import numpy as np
from maddpg_env.multiagent.core import World, Agent, Landmark
from maddpg_env.multiagent.scenario import BaseScenario

epsilon = 1e-3


def bow_s(d, radius):  # caculate the area of the bow give the chord and radius
    central_angle = np.arccos(d / (radius + epsilon)) * 2
    sector_area = 1 / 2 * central_angle * radius ** 2
    triangle_area = d * radius * np.sin(central_angle / 2)
    return sector_area - triangle_area


def coverae_s(entity1, entity2):  # the entity2 must be agent,entity1 could either be agent or landmark
    pos1, pos2 = entity1.state.p_pos, entity2.state.p_pos
    dist = np.sqrt(np.sum(np.square(pos1[:-1] - pos2[:-1])))
    # print('pos1:{}, pos2:{}'.format(pos1,pos2))
    # print('dist',dist)
    if isinstance(entity1, Agent):
        radius1 = pos1[::-1][0] * np.sin(entity1.coverage_theta)
    else:
        radius1 = entity1.size
    # print('radius1',radius1)
    radius2 = pos2[::-1][0] * np.sin(entity2.coverage_theta)
    # print('radius2',radius2)
    if dist > (radius1 + radius2):
        return 0
    elif dist + min(radius1, radius2) < max(radius1, radius2):
        return np.pi * min(radius1, radius2) ** 2
    else:
        x = (radius1 ** 2 + dist ** 2 - radius2 ** 2) / (2 * dist)
        y = dist - x
        # print('x/radius1:{},y/radius2:{}'.format(x/radius1
        #                                      ,y/radius2))
        return bow_s(x, radius1) + bow_s(y, radius2)


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 3
        world.dim_p = 3
        world_agents = 2
        num_landmarks = 1
        world.collaborative = True  # add attribute to the world
        world.agents = [Agent() for _ in range(world_agents)]

        for i, agent in enumerate(world.agents):
            agent.coverage_theta = True
            agent.coverage_theta = np.pi / 4
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True  # can not send communication message
            agent.size = 0.1
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 3
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.threshould = True
        world.threshould = 13
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for agent in world.agents:
            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_pos = np.random.uniform([-2,-2,0],[2,2,8],size=3)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform([-2,-2,0],[2,2,0])
            landmark.state.p_vel = np.zeros(world.dim_p)
        # for i in range(int(np.floor(len(world.landmarks) / 4))):
        #     for j in range(4):
        #         agent = world.landmarks[i * 4 + j]
        #         agent.state.p_pos[0] = 3 - 2 * i
        #         agent.state.p_pos[1] = 2 * j - 2

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # def reward(self,agent,world):
    #     rew = 0
    #
    # def reward(self, agent, world):
    #     rew = 0
    #     pos1 = agent.state.p_pos
    #     radius = pos1[::-1][0] * np.sin(agent.coverage_theta)
    #     for landmark in world.landmarks:
    #         pos_landmark = landmark.state.p_pos[:-1]
    #         dist = np.sqrt(np.sum(np.square(pos1[:-1] - pos_landmark)))
    #         if dist < radius:
    #             rew += 1
    #     # rew = rew*np.exp(-agent.state.p_pos[::-1][0])
    #     #print('coverage area {}'.format(rew))
    #     index = world.agents.index(agent)
    #
    #     def overlap_landmark(agent1, agent2, landmark):
    #         dist1 = np.sqrt(np.sum(np.square(agent1.state.p_pos[:-1] - landmark.state.p_pos[:-1])))
    #         dist2 = np.sqrt(np.sum(np.square(agent2.state.p_pos[:-1] - landmark.state.p_pos[:-1])))
    #         return dist1 < agent1.state.p_pos[::-1][0] * np.sin(agent1.coverage_theta) \
    #                and dist2 < agent2.state.p_pos[::-1][0] * np.sin(agent1.coverage_theta)
    #     landmarks = []
    #     for i in range(index + 1, len(world.agents)):
    #         landmarks = [landmark for landmark in world.landmarks if overlap_landmark(agent, world.agents[i], landmark)]
    #     if rew > world.threshould and len(landmarks) < 3:
    #         rew = 1
    #     else:
    #         rew = 0
    #     if agent.collide:
    #         for a in world.agents:
    #             if (a is not agent) and self.is_collision(a,
    #                                                       agent):  # if the agent is occur collision with one of the agent in the world
    #                 # then the reward is -1
    #                 rew -= 10
    #
    #     if agent.state.p_pos[::-1][0] < 0:
    #         rew -= 10
    #     return rew
    @ classmethod
    def reward(self, agent, world):
        rew = 0
        rew = coverae_s(entity1=world.landmarks[0], entity2 = agent)
        agent_index = world.agents.index(agent)
        for i in range(agent_index + 1, len(world.agents)):
            # print('coverage_area:{}'.format(rew))
            # print('ovler_loap{}:{}'.format(i,coverae_s(entity1=world.agents[i], entity2=agent)))
            rew -= 0.5 * coverae_s(entity1=world.agents[i], entity2=agent)
        if agent.state.p_pos[::-1][0] < 0:
            rew -= 5
        if agent.collide:
            for a in world.agents:
                if (a is not agent) and self.is_collision(a,
                                                          agent):  # if the agent is occur collision with one of the agent in the world
                    # then the reward is -1
                    rew -= 5
        return rew
        #return rew * np.exp(-agent.state.p_pos[::-1][0])

    def done(self, agent, world):
        # return False
        # # print(agent.state.p_pos)
        return agent.state.p_pos[world.dim_p - 1] < 0

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_size = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(
                entity.state.p_pos - agent.state.p_pos)  # the distance between the agent and the landmark
            entity_size.append(np.array([entity.size]))
        # entity colors
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:  continue
            comm.append(other.state.c)
            other_pos.append(
                other.state.p_pos - agent.state.p_pos)  # the realative position between the agent and other agent
        return np.concatenate(
             [agent.state.p_vel]+ [agent.state.p_pos]+entity_pos + other_pos + comm)
            