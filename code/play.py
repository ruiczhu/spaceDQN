import argparse
import pygame
from random import randint
from main import Game, Action, ResourceManager, Meteor, TARGET_REWARD
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
    
    agent = DQNAgent(state_size=35, action_size=len(Action))
    agent.load(args.model)
    agent.epsilon = 0  # 在测试时禁用随机探索
    
    fixed_dt = 1/50
    
    # 记录成功次数
    success_count = 0
    total_rewards = []
    
    for episode in range(args.episodes):
        player = game.reset()
        step = 0
        game.cumulative_reward = 0
        
        while step < 2000 and player.lives > 0 and not game.target_reached:
            dt = fixed_dt
            
            if step % 48 == 0: 
                Meteor(game.resources, (randint(0, game.WINDOW_WIDTH), -100), 
                        (game.all_sprites, game.meteor_sprites))
            
            state = player.get_state()
            if state is not None:
                state_values = list(state.values())
                action_idx = agent.act(state_values)
                action = Action(action_idx)
                player.take_action(action, dt)
            
            reward = game.update(dt)
            
            game.render()
            info_texts = [
                (f'Episode: {episode+1}/{args.episodes}', (255, 255, 0)),
                (f'Step: {step}', (240, 240, 240)),
                (f'Target Progress: {(game.cumulative_reward/TARGET_REWARD)*100:.1f}%', 
                 (240, 240, 240)),
                (f'Success Rate: {(success_count/max(1, episode))*100:.1f}%', 
                 (240, 240, 240)),
            ]
            
            # 如果达到目标，显示成功信息
            if game.target_reached:
                info_texts.append(
                    ('TARGET REACHED!', (0, 255, 0))
                )
            
            for i, (text, color) in enumerate(info_texts):
                surf = game.resources.fonts['main'].render(text, True, color)
                rect = surf.get_rect(topright=(game.WINDOW_WIDTH - 20, 80 + i * 40))
                game.display_surface.blit(surf, rect)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            step += 1
        
        # 更新统计信息
        if game.target_reached:
            success_count += 1
        
        total_rewards.append(game.cumulative_reward)
        avg_reward = sum(total_rewards) / len(total_rewards)
        
        print(f"Episode {episode+1}: "
              f"{'SUCCESS!' if game.target_reached else 'FAILED'} | "
              f"Score: {game.score} | "
              f"Steps: {step} | "
              f"Survival Time: {game.survival_time:.1f}s | "
              f"Final Reward: {game.cumulative_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f}")
    
    # 显示最终统计信息
    final_success_rate = (success_count / args.episodes) * 100
    final_avg_reward = sum(total_rewards) / len(total_rewards)
    print("\nFinal Statistics:")
    print(f"Success Rate: {final_success_rate:.1f}%")
    print(f"Average Reward: {final_avg_reward:.2f}")
    
    pygame.quit()

if __name__ == "__main__":
    play_game()