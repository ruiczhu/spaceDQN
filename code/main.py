import pygame
from os.path import join

from random import randint, uniform
from enum import Enum

class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    UP_LEFT = 5
    UP_RIGHT = 6
    DOWN_LEFT = 7
    DOWN_RIGHT = 8
    SHOOT = 9
    SHOOT_LEFT = 10
    SHOOT_RIGHT = 11
    SHOOT_UP = 12
    SHOOT_DOWN = 13
    SHOOT_UP_LEFT = 14
    SHOOT_UP_RIGHT = 15
    SHOOT_DOWN_LEFT = 16
    SHOOT_DOWN_RIGHT = 17

class ResourceManager:
    @staticmethod
    def init_display(fast_mode=True, enable_sound=False):
        pygame.init()
        if not fast_mode:
            pygame.display.set_mode((1280, 720))
        else:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
        if enable_sound:
            pygame.mixer.init()

    def __init__(self, fast_mode=True, enable_sound=False):
        self.images = {}
        self.fast_mode = fast_mode
        self.enable_sound = enable_sound
        self.fonts = {}
        if enable_sound:
            pygame.mixer.init()
            self.sounds = {}
        self.ensure_display_initialized()
        self.load_all()

    def ensure_display_initialized(self):
        if not pygame.get_init() or not pygame.display.get_surface():
            ResourceManager.init_display(self.fast_mode, self.enable_sound)

    def load_all(self):
        if not self.fast_mode:
            # 加载图像
            self.images.update({
                'star': pygame.image.load(join('images', 'star.png')).convert_alpha(),
                'meteor': pygame.image.load(join('images', 'meteor.png')).convert_alpha(),
                'laser': pygame.image.load(join('images', 'laser.png')).convert_alpha(),
                'player': pygame.image.load(join('images', 'player.png')).convert_alpha(),
                'explosion_frames': [
                    pygame.image.load(join('images', 'explosion', f'{i}.png')).convert_alpha()
                    for i in range(21)
                ]
            })

            # 只在启用声音时加载音效
            if self.enable_sound:
                self.sounds.update({
                    'laser': pygame.mixer.Sound(join('audio', 'laser.wav')),
                    'explosion': pygame.mixer.Sound(join('audio', 'explosion.wav')),
                    'music': pygame.mixer.Sound(join('audio', 'game_music.wav'))
                })

                # 设置音量
                self.sounds['laser'].set_volume(0.3)
                self.sounds['music'].set_volume(0.2)

            # 加载字体
            self.fonts.update({
                'main': pygame.font.Font(join('images', 'Oxanium-Bold.ttf'), 40)
            })
        else:
            # 快速模式：使用简单的surface替代图片
            surf = pygame.Surface((32, 32))
            surf.fill((255, 255, 255))
            self.images.update({
                'player': surf,
                'meteor': surf,
                'laser': pygame.Surface((4, 16)),
                'star': pygame.Surface((2, 2)),
                'explosion_frames': [surf]
            })

    def play_sound(self, sound_name):
        if self.enable_sound and hasattr(self, 'sounds'):
            self.sounds[sound_name].play()

class Game:
    def __init__(self, resources, fast_mode=True):
        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 720
        self.fast_mode = fast_mode

        # 添加时钟对象
        self.clock = pygame.time.Clock()

        if not fast_mode:
            self.display_surface = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        else:
            self.display_surface = pygame.Surface((1, 1))

        # 保存资源管理器引用
        self.resources = resources

        # 游戏状态
        self.score = 0
        self.current_reward = 0
        self.cumulative_reward = 0
        self.survival_time = 0
        self.target_reached = False  # 新增：是否达到目标的标志

        self.all_sprites = pygame.sprite.Group()
        self.meteor_sprites = pygame.sprite.Group()
        self.laser_sprites = pygame.sprite.Group()
        self.static_sprites = pygame.sprite.Group()

        # 初始化静态背景
        self.initialize_static_background()
        
        # 创建玩家
        self.create_player()

        # 设置陨石生成事件
        self.meteor_event = pygame.event.custom_type()
        pygame.time.set_timer(self.meteor_event, 500)

        self.ai_mode = True  # 添加AI模式标志

    def initialize_static_background(self):
        """初始化静态背景元素，只执行一次"""
        for i in range(20):
            Star(self.static_sprites, self.resources)
        self.all_sprites.add(self.static_sprites)

    def create_player(self, ai_controlled=False):
        """仅创建或重置玩家，不影响其他精灵"""
        # 移除旧的玩家精灵（如果存在）
        if hasattr(self, 'player'):
            self.player.kill()
        
        self.player = Player(self.all_sprites, self.resources)
        self.player.ai_controlled = ai_controlled
        self.player.game = self
        self.ai_mode = ai_controlled  # 更新AI模式标志
        return self.player

    def reset(self):
        # 重置游戏状态
        self.score = 0
        self.current_reward = 0
        self.cumulative_reward = 0
        self.survival_time = 0
        self.target_reached = False

        # 清除所有陨石和激光
        for sprite in self.meteor_sprites:
            sprite.kill()
        for sprite in self.laser_sprites:
            sprite.kill()

        # 重置玩家位置和状态
        if hasattr(self, 'player'):
            self.player.rect.center = (self.WINDOW_WIDTH / 2, self.WINDOW_HEIGHT / 2)
            self.player.lives = 2
            self.player.is_invulnerable = False
            self.player.can_shoot = True
            self.player.ai_controlled = True
        else:
            self.create_player(ai_controlled=True)

        return self.player

    def update(self, dt):
        self.current_reward = 0  # 确保每次更新开始时重置当前奖励
        self.survival_time += dt

        # 记录玩家的旧位置
        old_pos = pygame.Vector2(self.player.rect.center)

        # 更新所有精灵
        self.all_sprites.update(dt)

        # 检查玩家是否移动
        new_pos = pygame.Vector2(self.player.rect.center)
        if (round(new_pos.x) != round(old_pos.x)) or (round(new_pos.y) != round(old_pos.y)):
            self.current_reward += MOVEMENT_REWARD
        else:
            # 如果玩家没有移动，添加惩罚
            self.current_reward += NO_MOVEMENT_PENALTY * dt  # 根据时间步长调整惩罚

        game_continue = self.handle_collisions()

        # 添加基础生存奖励
        self.current_reward += SURVIVAL_REWARD * dt  # 根据时间步长调整奖励
        self.cumulative_reward += self.current_reward

        # 只在AI模式下检查目标奖励
        if self.ai_mode and self.cumulative_reward >= TARGET_REWARD:
            self.target_reached = True
            return False

        return game_continue

    def handle_collisions(self):
        """处理碰撞检测和更新分数与奖励"""
        collision_sprites = pygame.sprite.spritecollide(
            self.player, self.meteor_sprites, True, pygame.sprite.collide_mask
        )
        if collision_sprites and self.player.get_hit():
            self.current_reward += METEOR_HIT_PENALTY
            AnimatedExplosion(self.resources, self.player.rect.center, self.all_sprites)
            if self.player.lives <= 0:
                return False

        # 检查激光是否击中陨石
        for laser in self.laser_sprites:
            collided_sprites = pygame.sprite.spritecollide(laser, self.meteor_sprites, True)
            if collided_sprites:
                self.score += 100
                self.current_reward += METEOR_DESTROY_REWARD
                laser.kill()
                for meteor in collided_sprites:
                    AnimatedExplosion(self.resources, meteor.rect.center, self.all_sprites)

        # 添加存活奖励
        self.current_reward += SURVIVAL_REWARD
        self.cumulative_reward += self.current_reward
        return True

    def render(self):
        """优化的渲染函数"""
        if self.fast_mode:
            return
        
        self.display_surface.fill('#3a2e3f')
        
        # 先渲染静态精灵
        self.static_sprites.draw(self.display_surface)
        
        # 再渲染动态精灵
        for sprite in self.all_sprites:
            if sprite not in self.static_sprites:
                self.display_surface.blit(sprite.image, sprite.rect)
        
        self.display_game_info()
        self.display_rl_info()
        pygame.display.flip()

    def display_game_info(self):
        """显示基本游戏信息"""
        # 显示分数
        score_surf = self.resources.fonts['main'].render(
            f'Score: {self.score}', True, (240, 240, 240)
        )
        # FIXED: replaced get_frect with get_rect
        score_rect = score_surf.get_rect(topleft=(20, 20))
        self.display_surface.blit(score_surf, score_rect)

        # 显示生命值
        lives_surf = self.resources.fonts['main'].render(
            f'Lives: {self.player.lives}', True, (240, 240, 240)
        )
        # FIXED: replaced get_frect with get_rect
        lives_rect = lives_surf.get_rect(topright=(self.WINDOW_WIDTH - 20, 20))
        self.display_surface.blit(lives_surf, lives_rect)

        # 显示游戏状态
        status = 'INVULNERABLE' if self.player.is_invulnerable else 'NORMAL'
        status_surf = self.resources.fonts['main'].render(
            f'Status: {status}', True, (240, 240, 240)
        )
        # FIXED: replaced get_frect with get_rect
        status_rect = status_surf.get_rect(midtop=(self.WINDOW_WIDTH / 2, 20))
        self.display_surface.blit(status_surf, status_rect)

    def display_rl_info(self):
        """显示强化学习相关信息"""
        action_color = (255, 255, 0) if self.player.last_action != Action.NONE else (240, 240, 240)

        rl_info = [
            (f'Action: {self.player.last_action.name}', action_color),
            (
                f'Player Position: ({int(self.player.rect.centerx)}, {int(self.player.rect.centery)})',
                (240, 240, 240),
            ),
            (f'Cumulative Reward: {self.cumulative_reward:.1f}', (240, 240, 240)),
            (f'Survival Time: {self.survival_time:.1f}s', (240, 240, 240)),
        ]

        for i, (info, color) in enumerate(rl_info):
            info_surf = self.resources.fonts['main'].render(info, True, color)
            # FIXED: replaced get_frect with get_rect
            info_rect = info_surf.get_rect(topleft=(20, 80 + i * 40))
            self.display_surface.blit(info_surf, info_rect)

class Player(pygame.sprite.Sprite):
    def __init__(self, groups, resources):
        super().__init__(groups)
        self.resources = resources
        self.image = resources.images['player']

        # FIXED: replaced get_frect with get_rect
        self.rect = self.image.get_rect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2))
        self.direction = pygame.Vector2()
        self.speed = 500

        # cooldown
        self.can_shoot = True
        self.laser_shoot_time = 0
        self.cooldown_duration = 400

        # mask
        self.mask = pygame.mask.from_surface(self.image)

        self.lives = 2
        self.is_invulnerable = False
        self.invulnerable_timer = 0
        self.invulnerable_duration = 1000  # 2 seconds of invulnerability after hit

        self.ai_controlled = False
        self.last_action = Action.NONE
        self.game = None  # 将在创建时设置

        # 添加速度和方向追踪
        self.velocity = pygame.Vector2()
        self.last_velocity = pygame.Vector2()
        self.last_pos = pygame.Vector2(self.rect.center)

    def laser_timer(self):
        if not self.can_shoot:
            current_time = pygame.time.get_ticks()
            if current_time - self.laser_shoot_time >= self.cooldown_duration:
                self.can_shoot = True

    def get_hit(self):
        if not self.is_invulnerable:
            self.lives -= 1
            self.is_invulnerable = True
            self.invulnerable_timer = pygame.time.get_ticks()
            return True
        return False

    def get_state(self):
        """获取游戏状态，包含更多环境信息"""
        if not self.game:
            return None

        # 基础状态信息
        state = {
            'player_x': self.rect.centerx / self.game.WINDOW_WIDTH,
            'player_y': self.rect.centery / self.game.WINDOW_HEIGHT,
            'player_vx': self.velocity.x / self.speed,  # 归一化速度
            'player_vy': self.velocity.y / self.speed,
            'player_direction_x': self.direction.x,
            'player_direction_y': self.direction.y,
            'lives': self.lives / 3.0,
            'can_shoot': float(self.can_shoot),
            'is_invulnerable': float(self.is_invulnerable),
            'laser_cooldown': (pygame.time.get_ticks() - self.laser_shoot_time) / self.cooldown_duration,
            'active_lasers': len(self.game.laser_sprites) / 10.0,  # 归一化激光数量
        }

        # 收集陨石信息（修改后的逻辑）
        meteor_info = []
        for meteor in self.game.meteor_sprites:
            # 只关注在玩家上方的陨石
            if meteor.rect.centery <= self.rect.centery:
                dist = pygame.Vector2(self.rect.center).distance_to(pygame.Vector2(meteor.rect.center))
                rel_velocity = pygame.Vector2(meteor.direction) * meteor.speed
                # 计算陨石到玩家的水平距离
                horizontal_dist = abs(meteor.rect.centerx - self.rect.centerx)
                # 计算陨石到玩家的垂直距离
                vertical_dist = self.rect.centery - meteor.rect.centery  # 正值表示陨石在上方
                
                meteor_info.append({
                    'dist': dist,
                    'meteor': meteor,
                    'rel_velocity': rel_velocity,
                    'horizontal_dist': horizontal_dist,
                    'vertical_dist': vertical_dist
                })

        # 首先按垂直距离排序，优先考虑最近的威胁
        meteor_info.sort(key=lambda x: x['vertical_dist'])
        # 在相近垂直距离的情况下，考虑水平距离
        nearest_meteors = meteor_info[:10]

        # 填充虚拟陨石（如果收集到的陨石少于10个）
        while len(nearest_meteors) < 10:
            nearest_meteors.append({
                'dist': float('inf'),
                'meteor': None,
                'rel_velocity': pygame.Vector2(),
                'horizontal_dist': float('inf'),
                'vertical_dist': float('inf')
            })

        # 添加陨石信息（包括相对位置和速度）
        for i, meteor_data in enumerate(nearest_meteors):
            if meteor_data['meteor'] is not None:
                meteor = meteor_data['meteor']
                rel_velocity = meteor_data['rel_velocity']
                
                # 归一化位置和距离信息
                state.update({
                    f'meteor_{i}_x': meteor.rect.centerx / self.game.WINDOW_WIDTH,
                    f'meteor_{i}_y': meteor.rect.centery / self.game.WINDOW_HEIGHT,
                    f'meteor_{i}_dist': min(meteor_data['dist'] / self.game.WINDOW_WIDTH, 1.0),
                    f'meteor_{i}_h_dist': min(meteor_data['horizontal_dist'] / self.game.WINDOW_WIDTH, 1.0),
                    f'meteor_{i}_v_dist': min(meteor_data['vertical_dist'] / self.game.WINDOW_HEIGHT, 1.0),
                    f'meteor_{i}_vx': rel_velocity.x / 500.0,
                    f'meteor_{i}_vy': rel_velocity.y / 500.0,
                })
            else:
                # 对于虚拟陨石，使用默认值
                state.update({
                    f'meteor_{i}_x': 1.0,
                    f'meteor_{i}_y': 0.0,
                    f'meteor_{i}_dist': 1.0,
                    f'meteor_{i}_h_dist': 1.0,
                    f'meteor_{i}_v_dist': 1.0,
                    f'meteor_{i}_vx': 0.0,
                    f'meteor_{i}_vy': 0.0,
                })

        # 收集最近的3个激光信息
        laser_info = []
        for laser in self.game.laser_sprites:
            dist = pygame.Vector2(self.rect.center).distance_to(pygame.Vector2(laser.rect.center))
            laser_info.append((dist, laser))

        laser_info.sort(key=lambda x: x[0])
        nearest_lasers = laser_info[:3]

        # 填充虚拟激光
        while len(nearest_lasers) < 3:
            nearest_lasers.append((float('inf'), None))

        # 添加激光信息
        for i, (dist, laser) in enumerate(nearest_lasers):
            if laser is not None:
                state.update({
                    f'laser_{i}_x': laser.rect.centerx / self.game.WINDOW_WIDTH,
                    f'laser_{i}_y': laser.rect.centery / self.game.WINDOW_HEIGHT,
                    f'laser_{i}_dist': min(dist / self.game.WINDOW_WIDTH, 1.0),
                })
            else:
                state.update({
                    f'laser_{i}_x': 0.0,
                    f'laser_{i}_y': 0.0,
                    f'laser_{i}_dist': 1.0,
                })

        return state

    def take_action(self, action, dt):
        global CURRENT_ACTION, CURRENT_REWARD
        CURRENT_ACTION = action
        self.last_action = action

        # 重置方向
        self.direction.x = 0
        self.direction.y = 0
        should_shoot = False

        # 处理移动部分
        if action in [Action.LEFT, Action.UP_LEFT, Action.DOWN_LEFT, 
                     Action.SHOOT_LEFT, Action.SHOOT_UP_LEFT, Action.SHOOT_DOWN_LEFT]:
            self.direction.x = -1
        if action in [Action.RIGHT, Action.UP_RIGHT, Action.DOWN_RIGHT,
                     Action.SHOOT_RIGHT, Action.SHOOT_UP_RIGHT, Action.SHOOT_DOWN_RIGHT]:
            self.direction.x = 1
        if action in [Action.UP, Action.UP_LEFT, Action.UP_RIGHT,
                     Action.SHOOT_UP, Action.SHOOT_UP_LEFT, Action.SHOOT_UP_RIGHT]:
            self.direction.y = -1
        if action in [Action.DOWN, Action.DOWN_LEFT, Action.DOWN_RIGHT,
                     Action.SHOOT_DOWN, Action.SHOOT_DOWN_LEFT, Action.SHOOT_DOWN_RIGHT]:
            self.direction.y = 1

        # 处理射击部分
        if action in [Action.SHOOT, Action.SHOOT_LEFT, Action.SHOOT_RIGHT,
                     Action.SHOOT_UP, Action.SHOOT_DOWN, Action.SHOOT_UP_LEFT,
                     Action.SHOOT_UP_RIGHT, Action.SHOOT_DOWN_LEFT, Action.SHOOT_DOWN_RIGHT]:
            should_shoot = True

        if self.direction.length() > 0:
            self.direction = self.direction.normalize()

        new_x = self.rect.centerx + self.direction.x * self.speed * dt
        new_y = self.rect.centery + self.direction.y * self.speed * dt

        # 边界检查
        new_x = max(self.rect.width / 2, min(WINDOW_WIDTH - self.rect.width / 2, new_x))
        new_y = max(self.rect.height / 2, min(WINDOW_HEIGHT - self.rect.height / 2, new_y))

        self.rect.centerx = new_x
        self.rect.centery = new_y

        if should_shoot and self.can_shoot:
            Laser(
                self.resources,
                self.rect.midtop,
                (self.game.all_sprites, self.game.laser_sprites)
            )
            self.can_shoot = False
            self.laser_shoot_time = pygame.time.get_ticks()
            if hasattr(self.resources, 'sounds'):
                self.resources.play_sound('laser')
            # 添加射击奖励
            self.game.current_reward += SHOOT_REWARD

    def check_invulnerability(self):
        """检查并更新无敌状态"""
        if self.is_invulnerable:
            current_time = pygame.time.get_ticks()
            if current_time - self.invulnerable_timer >= self.invulnerable_duration:
                self.is_invulnerable = False

    def update(self, dt):
        global CURRENT_ACTION, CURRENT_REWARD

        if not self.ai_controlled:
            # 人类控制逻辑
            keys = pygame.key.get_pressed()
            self.handle_human_input(keys, dt)

        # 不管是AI还是人类控制，都需要处理这些状态
        self.laser_timer()
        self.check_invulnerability()

        # 更新速度信息
        current_pos = pygame.Vector2(self.rect.center)
        self.velocity = (current_pos - self.last_pos) / dt if dt > 0 else pygame.Vector2()
        self.last_pos = current_pos
        self.last_velocity = self.velocity

    def handle_human_input(self, keys, dt):
        global CURRENT_ACTION, CURRENT_REWARD

        # 组合动作检测
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        shooting = keys[pygame.K_SPACE]

        # 确定动作类型
        if shooting:
            if up:
                if left:
                    self.last_action = Action.SHOOT_UP_LEFT
                elif right:
                    self.last_action = Action.SHOOT_UP_RIGHT
                else:
                    self.last_action = Action.SHOOT_UP
            elif down:
                if left:
                    self.last_action = Action.SHOOT_DOWN_LEFT
                elif right:
                    self.last_action = Action.SHOOT_DOWN_RIGHT
                else:
                    self.last_action = Action.SHOOT_DOWN
            elif left:
                self.last_action = Action.SHOOT_LEFT
            elif right:
                self.last_action = Action.SHOOT_RIGHT
            else:
                self.last_action = Action.SHOOT
        else:
            if up:
                if left:
                    self.last_action = Action.UP_LEFT
                elif right:
                    self.last_action = Action.UP_RIGHT
                else:
                    self.last_action = Action.UP
            elif down:
                if left:
                    self.last_action = Action.DOWN_LEFT
                elif right:
                    self.last_action = Action.DOWN_RIGHT
                else:
                    self.last_action = Action.DOWN
            elif left:
                self.last_action = Action.LEFT
            elif right:
                self.last_action = Action.RIGHT
            else:
                self.last_action = Action.NONE

        CURRENT_ACTION = self.last_action

        # 转换为向量移动
        self.direction = pygame.Vector2()
        if left: self.direction.x -= 1
        if right: self.direction.x += 1
        if up: self.direction.y -= 1
        if down: self.direction.y += 1

        if self.direction.length() > 0:
            self.direction = self.direction.normalize()

        # 更新位置
        new_x = self.rect.centerx + self.direction.x * self.speed * dt
        new_y = self.rect.centery + self.direction.y * self.speed * dt

        # 边界检查
        new_x = max(self.rect.width / 2, min(WINDOW_WIDTH - self.rect.width / 2, new_x))
        new_y = max(self.rect.height / 2, min(WINDOW_HEIGHT - self.rect.height / 2, new_y))

        # 如果玩家确实移动了，扣除移动成本
        if (round(new_x) != round(self.rect.centerx)) or (round(new_y) != round(self.rect.centery)):
            CURRENT_REWARD += MOVEMENT_REWARD

        self.rect.centerx = new_x
        self.rect.centery = new_y

        # 射击处理
        if shooting and self.can_shoot:
            Laser(
                self.resources,
                self.rect.midtop,
                (self.game.all_sprites, self.game.laser_sprites)
            )
            self.can_shoot = False
            self.laser_shoot_time = pygame.time.get_ticks()
            if hasattr(self.resources, 'sounds'):
                self.resources.play_sound('laser')
            # 添加射击奖励
            CURRENT_REWARD += SHOOT_REWARD

class Star(pygame.sprite.Sprite):
    def __init__(self, groups, resources):
        super().__init__(groups)
        self.image = resources.images['star']
        # FIXED: replaced get_frect with get_rect
        self.rect = self.image.get_rect(
            center=(randint(0, WINDOW_WIDTH), randint(0, WINDOW_HEIGHT))
        )

class Laser(pygame.sprite.Sprite):
    def __init__(self, resources, pos, groups):
        super().__init__(groups)
        self.image = resources.images['laser']
        # FIXED: replaced get_frect with get_rect
        self.rect = self.image.get_rect(midbottom=pos)

    def update(self, dt):
        self.rect.centery -= 400 * dt
        if self.rect.bottom < 0:
            self.kill()

class Meteor(pygame.sprite.Sprite):
    def __init__(self, resources, pos, groups):
        super().__init__(groups)
        self.image = resources.images['meteor']
        # FIXED: replaced get_frect with get_rect
        self.rect = self.image.get_rect(center=pos)
        self.direction = pygame.Vector2(uniform(-0.5, 0.5), 1)
        self.speed = randint(200, 300)
        if not resources.fast_mode:
            self.rotation_speed = randint(40, 80)
            self.rotation = 0
            self.original_surf = self.image

        self.original_velocity = pygame.Vector2(self.direction) * self.speed

    def update(self, dt):
        self.rect.center += self.direction * self.speed * dt
        if hasattr(self, 'rotation_speed'):
            self.rotation += self.rotation_speed * dt
            self.image = pygame.transform.rotozoom(self.original_surf, self.rotation, 1)
            # FIXED: replaced get_frect with get_rect
            self.rect = self.image.get_rect(center=self.rect.center)

class AnimatedExplosion(pygame.sprite.Sprite):
    def __init__(self, resources, pos, groups):
        super().__init__(groups)
        if resources.fast_mode:
            self.kill()  # 快速模式直接移除爆炸效果
            return
        self.frames = resources.images['explosion_frames']
        self.frame_index = 0
        self.image = self.frames[self.frame_index]
        # FIXED: replaced get_frect with get_rect
        self.rect = self.image.get_rect(center=pos)
        if hasattr(resources, 'sounds'):
            resources.play_sound('explosion')

    def update(self, dt):
        self.frame_index += 20 * dt
        if self.frame_index < len(self.frames):
            self.image = self.frames[int(self.frame_index)]
        else:
            self.kill()

# 全局变量
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
GAME_SCORE = 0
CURRENT_ACTION = Action.NONE
CURRENT_REWARD = 0
CUMULATIVE_REWARD = 0
SURVIVAL_TIME = 0


SURVIVAL_REWARD = 0.01  # 每帧生存奖励
MOVEMENT_REWARD = 0.03  # 移动奖励
SHOOT_REWARD = 5.0     # 射击奖励
NO_MOVEMENT_PENALTY = -0.4  # 不移动惩罚（新增）
METEOR_HIT_PENALTY = -30.0
METEOR_DESTROY_REWARD = 5.0
TARGET_REWARD = 200.0  # 新增：目标奖励阈值

def main_game_loop():
    """人类控制的主游戏循环"""
    global running, GAME_SCORE, CURRENT_REWARD, CUMULATIVE_REWARD, SURVIVAL_TIME

    # 初始化显示和资源 - 人类模式下使用完整功能
    ResourceManager.init_display(fast_mode=False, enable_sound=True)  # 使用完整显示模式
    resources = ResourceManager(fast_mode=False, enable_sound=True)   # 加载所有资源，包括声音

    # 创建游戏实例，设置为人类控制模式
    game = Game(resources, fast_mode=False)
    game.create_player(ai_controlled=False)  # 确保设置为人类控制模式

    # 播放背景音乐
    if hasattr(resources, 'sounds'):
        resources.sounds['music'].play(loops=-1)

    # 游戏主循环
    running = True
    while running:
        dt = game.clock.tick(60) / 1000  # 限制帧率为60FPS
        CURRENT_REWARD = 0
        SURVIVAL_TIME += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # FIXED: handle meteor creation here
            if event.type == game.meteor_event:
                x, y = randint(0, game.WINDOW_WIDTH), randint(-200, -100)
                Meteor(game.resources, (x, y), (game.all_sprites, game.meteor_sprites))

            # Optional: if you want "just pressed" for spacebar, handle pygame.KEYDOWN:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Example of "just pressed" logic for shoot:
                if game.player.can_shoot:
                    Laser(
                        resources,
                        game.player.rect.midtop,
                        (game.all_sprites, game.laser_sprites)
                    )
                    game.player.can_shoot = False
                    game.player.laser_shoot_time = pygame.time.get_ticks()
                    if hasattr(resources, 'sounds'):
                        resources.play_sound('laser')

        if not game.update(dt):
            running = False

        game.render()

    pygame.quit()

if __name__ == "__main__":
    # 仅在直接运行 main.py 时，才执行人类控制模式
    main_game_loop()