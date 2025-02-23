import argparse
import pygame
from random import randint
from main import Game, Action, ResourceManager, Meteor
from dqn_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        help='path to the trained model file')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='number of episodes to play')
    return parser.parse_args()

def play_game():
    args = parse_args()
    
    ResourceManager.init_display(False)
    resources = ResourceManager(False)
    game = Game(resources, False)
    
    # 更新状态空间大小以匹配训练时的设置 (5个基础状态 + 10个陨石各3个状态)
    agent = DQNAgent(state_size=35, action_size=len(Action))
    agent.load(args.model)
    agent.epsilon = 0  # 在测试时使用较小的随机探索
    
    fixed_dt = 1/50  # 改为1/60，使移动更平滑
    clock = pygame.time.Clock()  # 添加时钟对象控制帧率
    
    for episode in range(args.episodes):
        player = game.reset()
        step = 0
        game.cumulative_reward = 0  # 确保每个episode开始时重置累积奖励
        
        while step < 1000 and player.lives > 0:
            # clock.tick(60)  # 限制帧率为60FPS
            dt = fixed_dt
            
            # 创建更多陨石以测试模型的性能
            if step % 48 == 0: 
                Meteor(game.resources, (randint(0, game.WINDOW_WIDTH), -100), 
                        (game.all_sprites, game.meteor_sprites))
            
            # 获取状态并执行动作
            state = player.get_state()
            if state is not None:
                state_values = list(state.values())
                action_idx = agent.act(state_values)
                action = Action(action_idx)
                player.take_action(action, dt)
            
            # 更新游戏状态
            reward = game.update(dt)
            
            # 渲染增强的状态信息
            game.render()
            info_texts = [
                (f'Episode: {episode+1}/{args.episodes}', (255, 255, 0)),
                (f'Step: {step}', (240, 240, 240)),
            ]
            
            for i, (text, color) in enumerate(info_texts):
                surf = game.resources.fonts['main'].render(text, True, color)
                rect = surf.get_rect(topright=(game.WINDOW_WIDTH - 20, 80 + i * 40))
                game.display_surface.blit(surf, rect)
            
            pygame.display.flip()
            
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            step += 1
        
        print(f"Episode {episode+1}: "
              f"Score: {game.score}, "
              f"Steps: {step}, "
              f"Survival Time: {game.survival_time:.1f}s, "
              f"Final Reward: {game.cumulative_reward:.2f}")
    
    pygame.quit()

if __name__ == "__main__":
    play_game()