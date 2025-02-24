from random import randint
import random
from main import TARGET_REWARD, Game, Action, ResourceManager, Meteor
import pygame
from dqn_agent import DQNAgent
import numpy as np
import argparse
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', 
                       help='enable rendering and visual effects (slower)')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='number of episodes to train')
    return parser.parse_args()

def plot_training_data(episode_rewards, losses, epsilon_values, window_size=100):
    """绘制训练数据图表"""
    plt.figure(figsize=(15, 10))
    
    # 绘制累积奖励
    plt.subplot(311)
    plt.plot(episode_rewards)
    plt.plot(np.convolve(episode_rewards, 
            np.ones(window_size)/window_size, mode='valid'), 
            label=f'Moving Average ({window_size})')
    plt.title('Cumulative Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # 绘制损失值
    plt.subplot(312)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    
    # 绘制Epsilon值
    plt.subplot(313)
    plt.plot(epsilon_values)
    plt.title('Epsilon Value')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()

def train_agent():
    args = parse_args()
    fast_mode = not args.render
    
    # 初始化时就确定是否需要渲染
    ResourceManager.init_display(fast_mode, enable_sound=False)
    resources = ResourceManager(fast_mode, enable_sound=False)
    game = Game(resources, fast_mode)
    
    # 更新为正确的状态空间大小
    state_size = 90  # 11(基础) + 70(陨石) + 9(激光)
    agent = DQNAgent(state_size=state_size, action_size=len(Action))
    batch_size = 64
    fixed_dt = 1/50
    
    # 增加预填充数量到batch_size的20倍
    min_samples = batch_size * 20
    print(f"预填充经验回放缓冲区 (目标: {min_samples} 样本)...")
    
    attempts_without_reset = 0
    total_samples = 0
    meteor_spawn_rate = 8  # 更频繁的陨石生成
    max_steps_per_episode = 500  # 每个预填充回合的最大步数
    
    while total_samples < min_samples:
        player = game.reset()
        state = player.get_state()
        
        if state is None:
            attempts_without_reset += 1
            if attempts_without_reset > 2:  # 连续3次失败后重新初始化游戏
                game = Game(resources, fast_mode)
                attempts_without_reset = 0
            continue
        
        attempts_without_reset = 0
        steps_in_episode = 0
        
        # 在每个回合开始时生成多个陨石
        for _ in range(10):
            Meteor(game.resources, 
                  (randint(0, game.WINDOW_WIDTH), -100),
                  (game.all_sprites, game.meteor_sprites))
        
        while steps_in_episode < max_steps_per_episode and player.lives > 0:
            # 增加陨石生成频率
            if steps_in_episode % meteor_spawn_rate == 0:
                Meteor(game.resources, 
                      (randint(0, game.WINDOW_WIDTH), -100),
                      (game.all_sprites, game.meteor_sprites))
            
            # 修复：先选择动作
            current_action = None
            if random.random() < 0.5:
                actions = []
                if random.random() < 0.5:
                    actions.append(random.choice([Action.LEFT, Action.RIGHT]))
                if random.random() < 0.5:
                    actions.append(random.choice([Action.UP, Action.DOWN]))
                if random.random() < 0.5:
                    actions.append(Action.SHOOT)
                
                for action in actions:
                    current_action = action  # 记录最后一个动作
                    player.take_action(action, fixed_dt)
            else:
                current_action = Action(random.randrange(len(Action)))
                player.take_action(current_action, fixed_dt)
                
            # 定期强制移动
            if steps_in_episode % 50 == 0:
                current_action = random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])
                player.take_action(current_action, fixed_dt)
            
            reward = game.update(fixed_dt)
            next_state = player.get_state()
            
            if next_state is not None and current_action is not None:  # 确保动作已定义
                total_samples += 1
                agent.remember(list(state.values()), current_action.value, reward, 
                             list(next_state.values()), player.lives <= 0,
                             training=False)
                state = next_state
                
                if total_samples % 100 == 0:
                    print(f"\r预填充进度: {total_samples}/{min_samples} "
                          f"({(total_samples/min_samples)*100:.1f}%)", end="")
            
            steps_in_episode += 1
    
    print("\n预填充完成!")
    agent.reset_epsilon()
    
    print("\n开始训练...")
    cumulative_rewards = []
    
    episode_rewards = []
    losses = []
    epsilon_values = []
    training_start_time = time.time()
    
    if args.episodes == 1000:
        if args.render:
            args.episodes = 200
        else:
            args.episodes = 2000
    
    # 在训练循环中不需要修改，因为reset已经优化
    for episode in range(args.episodes):
        player = game.reset()
        step = 0
        game.cumulative_reward = 0
        episode_losses = []
        
        while player.lives > 0 and not game.target_reached:
            dt = fixed_dt
            
            if step % 24 == 0:
                Meteor(game.resources, (randint(0, game.WINDOW_WIDTH), -100), 
                      (game.all_sprites, game.meteor_sprites))
            
            state = player.get_state()
            if state is not None:
                state_values = list(state.values())
                action_idx = agent.act(state_values)
                action = Action(action_idx)
                player.take_action(action, dt)
            
            reward = game.update(dt)
            next_state = player.get_state()
            
            if next_state is not None:
                # 更新完成条件：生命值为0或达到目标奖励
                done = player.lives <= 0 or game.target_reached
                agent.remember(state_values, action_idx, reward, 
                             list(next_state.values()), done)
                
                loss = agent.replay(batch_size)
                if loss is not None:
                    episode_losses.append(loss)
            
            if args.render:
                game.render()
                info_texts = [
                    (f'Episode: {episode+1}/{args.episodes}', (255, 255, 0)),
                    (f'Step Reward: {reward:.2f}', (240, 240, 240)),
                    (f'Epsilon: {agent.epsilon:.2f}', (240, 240, 240)),
                    (f'Loss: {np.mean(episode_losses) if episode_losses else "N/A"}', (240, 240, 240)),
                    (f'Memory: {len(agent.memory)}/{agent.min_experiences}', (240, 240, 240))
                ]
                
                info_texts.extend([
                    (f'Target Progress: {(game.cumulative_reward/TARGET_REWARD)*100:.1f}%', 
                     (240, 240, 240)),
                    (f'Target Reached: {game.target_reached}', 
                     (0, 255, 0) if game.target_reached else (240, 240, 240))
                ])
                
                for i, (text, color) in enumerate(info_texts):
                    surf = game.resources.fonts['main'].render(text, True, color)
                    rect = surf.get_rect(topright=(game.WINDOW_WIDTH - 20, 80 + i * 40))
                    game.display_surface.blit(surf, rect)
                
                pygame.display.flip()
            
            if game.target_reached:
                print(f"\nTarget reward reached in episode {episode+1}!")
                break
            
            step += 1
        
        # 在episode结束时更新epsilon
        agent.update_epsilon()
        
        episode_rewards.append(game.cumulative_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(None)
        
        epsilon_values.append(agent.epsilon)
        
        cumulative_rewards.append(game.cumulative_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(cumulative_rewards[-10:])
            valid_losses = [l for l in losses[-100:] if l is not None]
            avg_loss = np.mean(valid_losses) if valid_losses else float('nan')
            elapsed_time = time.time() - training_start_time
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            loss_str = f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "N/A"
            status_message = (
                f"Episode: {episode+1}/{args.episodes}, "
                f"Steps: {step}, "
                f"Cumulative Reward: {game.cumulative_reward:.2f}, "
                f"Average Reward: {avg_reward:.2f}, "
                f"Loss: {loss_str}, "
                f"Epsilon: {agent.epsilon:.4f}, "
                f"Training Time: {elapsed_time:.1f}s, "
                f"Learning Rate: {current_lr:.6f}"
            )
            print(status_message)
            
            plot_training_data(episode_rewards, losses, epsilon_values)
        
        if (episode + 1) % 100 == 0:
            agent.save(f'models/dqn_model_ep{episode+1}.pth')
            np.save('training_data.npz', {
                'rewards': episode_rewards,
                'losses': losses,
                'epsilons': epsilon_values
            })
    
    plot_training_data(episode_rewards, losses, epsilon_values)
    pygame.quit()

if __name__ == "__main__":
    train_agent()
