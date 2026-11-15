"""
Redis分组队列Lua脚本

提供原子性操作的Lua脚本，确保队列状态的一致性。
"""

# 通用rebalance函数定义
REBALANCE_FUNCTION = """
-- rebalance分区函数
local function rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
    -- 获取所有活跃的owner
    local active_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
    local owner_count = #active_owners
    
    if owner_count == 0 then
        return {0, {}}
    end
    
    -- 清理所有owner的queue_list
    for _, owner_id in ipairs(active_owners) do
        local queue_list_key = queue_list_prefix .. owner_id
        redis.call('DEL', queue_list_key)
    end
    
    -- 平均分配分区
    local partitions_per_owner = math.floor(total_partitions / owner_count)
    local extra_partitions = total_partitions % owner_count
    
    -- 使用扁平数组格式返回分配结果，便于Redis客户端正确转换
    local assigned_partitions_flat = {}
    local partition_index = 1
    
    for i, owner_id in ipairs(active_owners) do
        local queue_list_key = queue_list_prefix .. owner_id
        local partitions_for_this_owner = partitions_per_owner
        
        -- 前extra_partitions个owner多分配一个分区
        if i <= extra_partitions then
            partitions_for_this_owner = partitions_for_this_owner + 1
        end
        
        local owner_partitions = {}
        for j = 1, partitions_for_this_owner do
            local partition_name = string.format("%03d", partition_index)
            redis.call('LPUSH', queue_list_key, partition_name)
            table.insert(owner_partitions, partition_name)
            partition_index = partition_index + 1
        end
        
        -- 设置过期时间
        redis.call('EXPIRE', queue_list_key, owner_expire)
        
        -- 将owner_id和分区列表添加到扁平数组中
        table.insert(assigned_partitions_flat, owner_id)
        table.insert(assigned_partitions_flat, owner_partitions)
    end
    
    return {owner_count, assigned_partitions_flat}
end
"""

# 添加消息到队列的Lua脚本
ENQUEUE_SCRIPT = """
-- 参数：
-- KEYS[1]: 队列键 (zset)
-- KEYS[2]: 总数计数器键
-- ARGV[1]: 消息内容 (支持JSON字符串或BSON二进制数据)
-- ARGV[2]: 排序分数
-- ARGV[3]: 队列过期时间 (秒)
-- ARGV[4]: 活动时间过期时间 (秒，7天)
-- ARGV[5]: 最大总数限制

local queue_key = KEYS[1]
local counter_key = KEYS[2]
local message = ARGV[1]
local score = tonumber(ARGV[2])
local queue_expire = tonumber(ARGV[3])
local activity_expire = tonumber(ARGV[4])
local max_total = tonumber(ARGV[5])

-- 检查总数限制
local current_count = tonumber(redis.call('GET', counter_key) or '0')
if current_count >= max_total then
    return {0, current_count, "超过最大总数限制"}
end

-- 添加消息到队列
local added = redis.call('ZADD', queue_key, score, message)
if added == 1 then
    -- 更新队列过期时间
    redis.call('EXPIRE', queue_key, queue_expire)
    
    -- 增加总数计数
    local new_count = redis.call('INCR', counter_key)
    redis.call('EXPIRE', counter_key, activity_expire)
    
    return {1, new_count, "添加成功"}
else
    return {0, current_count, "消息已存在"}
end
"""

# Rebalance重新分区的Lua脚本
REBALANCE_PARTITIONS_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- ARGV[1]: 分区总数
-- ARGV[2]: owner过期时间（秒，默认1小时）

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local total_partitions = tonumber(ARGV[1])
local owner_expire = tonumber(ARGV[2])

-- 调用rebalance函数
return rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
"""

# 加入消费者的Lua脚本
JOIN_CONSUMER_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- ARGV[1]: owner_id
-- ARGV[2]: 当前时间戳
-- ARGV[3]: owner过期时间（秒，默认1小时）
-- ARGV[4]: 分区总数

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local owner_id = ARGV[1]
local current_time = tonumber(ARGV[2])
local owner_expire = tonumber(ARGV[3])
local total_partitions = tonumber(ARGV[4])

-- 加入owner_activate_time_zset
redis.call('ZADD', owner_zset_key, current_time, owner_id)
redis.call('EXPIRE', owner_zset_key, owner_expire)

-- 调用rebalance函数
return rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
"""

# 消费者退出的Lua脚本
EXIT_CONSUMER_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- ARGV[1]: owner_id
-- ARGV[2]: owner过期时间（秒，默认1小时）
-- ARGV[3]: 分区总数

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local owner_id = ARGV[1]
local owner_expire = tonumber(ARGV[2])
local total_partitions = tonumber(ARGV[3])

-- 从owner_activate_time_zset删除
redis.call('ZREM', owner_zset_key, owner_id)

-- 删除对应的queue_list
local queue_list_key = queue_list_prefix .. owner_id
redis.call('DEL', queue_list_key)

-- 检查是否还有剩余owner，如果有则调用rebalance函数
local remaining_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
if #remaining_owners == 0 then
    return {0, {}}
end

-- 调用rebalance函数
return rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
"""

# 消费者保活的Lua脚本
KEEPALIVE_CONSUMER_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- ARGV[1]: owner_id
-- ARGV[2]: 当前时间戳
-- ARGV[3]: owner过期时间（秒，默认1小时）

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local owner_id = ARGV[1]
local current_time = tonumber(ARGV[2])
local owner_expire = tonumber(ARGV[3])

-- 更新owner_activate_time_zset的时间
local updated = redis.call('ZADD', owner_zset_key, current_time, owner_id)
redis.call('EXPIRE', owner_zset_key, owner_expire)

-- 续期对应的queue_list
local queue_list_key = queue_list_prefix .. owner_id
local exists = redis.call('EXISTS', queue_list_key)
if exists == 1 then
    redis.call('EXPIRE', queue_list_key, owner_expire)
    return 1
else
    return 0
end
"""

# 定期清理不活跃owner的Lua脚本
CLEANUP_INACTIVE_OWNERS_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- KEYS[3]: queue_prefix (用于构建分区队列键)
-- KEYS[4]: counter_key (消息总数计数器键)
-- ARGV[1]: 不活跃阈值时间戳（5分钟前）
-- ARGV[2]: 当前时间戳
-- ARGV[3]: owner过期时间（秒，默认1小时）
-- ARGV[4]: 分区总数

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local queue_prefix = KEYS[3]
local counter_key = KEYS[4]
local inactive_threshold = tonumber(ARGV[1])
local current_time = tonumber(ARGV[2])
local owner_expire = tonumber(ARGV[3])
local total_partitions = tonumber(ARGV[4])

-- 获取所有不活跃的owner
local inactive_owners = redis.call('ZRANGEBYSCORE', owner_zset_key, 0, inactive_threshold)
local cleaned_count = 0

-- 清理不活跃的owner
for _, owner_id in ipairs(inactive_owners) do
    -- 从zset删除
    redis.call('ZREM', owner_zset_key, owner_id)
    
    -- 删除对应的queue_list
    local queue_list_key = queue_list_prefix .. owner_id
    redis.call('DEL', queue_list_key)
    
    cleaned_count = cleaned_count + 1
end

-- 无论是否有清理，都重新统计counter_key确保数据一致性
local total_messages = 0
for i = 1, total_partitions do
    local partition_name = string.format("%03d", i)
    local queue_key = queue_prefix .. partition_name
    local queue_size = redis.call('ZCARD', queue_key)
    total_messages = total_messages + queue_size
end
redis.call('SET', counter_key, total_messages)

-- 如果有清理，需要rebalance
local need_rebalance = cleaned_count > 0
if not need_rebalance then
    return {cleaned_count, 0, {}}
end

-- 检查是否还有剩余owner
local remaining_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
if #remaining_owners == 0 then
    return {cleaned_count, 0, {}}
end

-- 调用rebalance函数
local owner_count, assigned_partitions = unpack(rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire))
return {cleaned_count, owner_count, assigned_partitions}
"""

# 强制清理和重置的Lua脚本（支持可选清库）
FORCE_CLEANUP_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- KEYS[3]: queue_prefix (用于构建分区队列键)
-- KEYS[4]: counter_key (消息总数计数器键)
-- ARGV[1]: 分区总数
-- ARGV[2]: purge_all 标志（"1" 清空所有分区队列并置0；否则仅重算计数器）

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local queue_prefix = KEYS[3]
local counter_key = KEYS[4]
local total_partitions = tonumber(ARGV[1])
local purge_all = ARGV[2]

-- 获取所有owner
local all_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
local cleaned_count = 0

-- 删除所有owner的queue_list
for _, owner_id in ipairs(all_owners) do
    local queue_list_key = queue_list_prefix .. owner_id
    redis.call('DEL', queue_list_key)
    cleaned_count = cleaned_count + 1
end

-- 删除owner_activate_time_zset
redis.call('DEL', owner_zset_key)

if purge_all == '1' then
    -- 清空所有分区队列并将计数器置0
    for i = 1, total_partitions do
        local partition_name = string.format("%03d", i)
        local queue_key = queue_prefix .. partition_name
        redis.call('DEL', queue_key)
    end
    redis.call('SET', counter_key, 0)
    return total_partitions
else
    -- 仅重算计数器
    local total_messages = 0
    for i = 1, total_partitions do
        local partition_name = string.format("%03d", i)
        local queue_key = queue_prefix .. partition_name
        local queue_size = redis.call('ZCARD', queue_key)
        total_messages = total_messages + queue_size
    end
    redis.call('SET', counter_key, total_messages)
    return cleaned_count
end
"""

# 获取消息的Lua脚本（遍历所有分区各自尝试获取一个）
GET_MESSAGES_SCRIPT = """
-- 参数：
-- KEYS[1]: owner_activate_time_zset 键
-- KEYS[2]: queue_list_prefix (用于构建每个owner的queue_list键)
-- KEYS[3]: queue_prefix (用于构建分区队列键)
-- KEYS[4]: counter_key (消息总数计数器键)
-- ARGV[1]: owner_id
-- ARGV[2]: owner过期时间（秒，默认1小时）
-- ARGV[3]: score差值阈值（毫秒）
-- ARGV[4]: 当前score（用于空队列时的threshold比较）

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local queue_prefix = KEYS[3]
local counter_key = KEYS[4]
local owner_id = ARGV[1]
local owner_expire = tonumber(ARGV[2])
local score_threshold = tonumber(ARGV[3])
local current_score = tonumber(ARGV[4])

-- 检查owner是否存在于zset中
local owner_score = redis.call('ZSCORE', owner_zset_key, owner_id)
if not owner_score then
    -- owner不存在，需要加入消费者
    return {"JOIN_REQUIRED", {}}
end

-- 检查queue_list是否存在
local queue_list_key = queue_list_prefix .. owner_id
local queue_list_exists = redis.call('EXISTS', queue_list_key)
if queue_list_exists == 0 then
    -- queue_list不存在，需要加入消费者
    return {"JOIN_REQUIRED", {}}
end

-- 获取owner的队列列表
local owner_queues = redis.call('LRANGE', queue_list_key, 0, -1)
if #owner_queues == 0 then
    return {"NO_QUEUES", {}}
end

local messages = {}
local messages_consumed = 0

-- 遍历所有分区，每个尝试获取1个消息
for _, partition in ipairs(owner_queues) do
    local queue_key = queue_prefix .. partition
    
    -- 检查队列是否有消息
    local queue_size = redis.call('ZCARD', queue_key)
    if queue_size > 0 then
        -- 获取最早消息的score
        local min_result = redis.call('ZRANGE', queue_key, 0, 0, 'WITHSCORES')
        
        -- 直接比较最早消息与当前score的差值
        if #min_result >= 2 then
            local earliest_message_score = tonumber(min_result[2])
            -- 检查最早消息score与当前score的差值
            if (current_score - earliest_message_score) >= score_threshold then
                -- 获取最早的消息（直接删除）
                local popped = redis.call('ZPOPMIN', queue_key)
                if #popped >= 2 then
                    table.insert(messages, popped[1])  -- 只返回消息内容
                    messages_consumed = messages_consumed + 1
                end
            end
        end
    end
end

-- 如果有消息被消费，减少counter_key计数
if messages_consumed > 0 then
    local new_count = redis.call('DECRBY', counter_key, messages_consumed)
    -- 确保计数不会变成负数
    if new_count < 0 then
        redis.call('SET', counter_key, 0)
    end
end

-- 续期queue_list
redis.call('EXPIRE', queue_list_key, owner_expire)

return {"SUCCESS", messages}
"""

# 获取队列统计信息的Lua脚本
GET_QUEUE_STATS_SCRIPT = """
-- 参数：
-- KEYS[1]: 队列键 (zset)
-- KEYS[2]: 总数计数器键

local queue_key = KEYS[1]
local counter_key = KEYS[2]

-- 获取队列大小
local queue_size = redis.call('ZCARD', queue_key)

-- 获取总数
local total_count = tonumber(redis.call('GET', counter_key) or '0')

-- 获取队列的分数范围
local min_max = {}
if queue_size > 0 then
    local min_result = redis.call('ZRANGE', queue_key, 0, 0, 'WITHSCORES')
    local max_result = redis.call('ZRANGE', queue_key, -1, -1, 'WITHSCORES')
    if #min_result >= 2 then
        min_max.min_score = tonumber(min_result[2])
    end
    if #max_result >= 2 then
        min_max.max_score = tonumber(max_result[2])
    end
end

return {
    queue_size,
    total_count,
    min_max.min_score or 0,
    min_max.max_score or 0
}
"""

# 批量获取所有分区统计信息的Lua脚本
GET_ALL_PARTITIONS_STATS_SCRIPT = """
-- 参数：
-- KEYS[1]: queue_prefix (用于构建分区队列键)
-- KEYS[2]: 总数计数器键
-- ARGV[1]: 分区总数

local queue_prefix = KEYS[1]
local counter_key = KEYS[2]
local total_partitions = tonumber(ARGV[1])

-- 获取总数
local total_count = tonumber(redis.call('GET', counter_key) or '0')

-- 存储所有分区的统计信息
local partition_stats = {}
local total_messages_in_queues = 0
local global_min_score = nil
local global_max_score = nil

-- 遍历所有分区
for i = 1, total_partitions do
    local partition_name = string.format("%03d", i)
    local queue_key = queue_prefix .. partition_name
    
    -- 获取队列大小
    local queue_size = redis.call('ZCARD', queue_key)
    total_messages_in_queues = total_messages_in_queues + queue_size
    
    local min_score = 0
    local max_score = 0
    
    if queue_size > 0 then
        -- 获取最小和最大score
        local min_result = redis.call('ZRANGE', queue_key, 0, 0, 'WITHSCORES')
        local max_result = redis.call('ZRANGE', queue_key, -1, -1, 'WITHSCORES')
        
        if #min_result >= 2 then
            min_score = tonumber(min_result[2])
            if global_min_score == nil or min_score < global_min_score then
                global_min_score = min_score
            end
        end
        
        if #max_result >= 2 then
            max_score = tonumber(max_result[2])
            if global_max_score == nil or max_score > global_max_score then
                global_max_score = max_score
            end
        end
    end
    
    -- 存储分区统计信息（扁平数组格式）
    table.insert(partition_stats, partition_name)
    table.insert(partition_stats, queue_size)
    table.insert(partition_stats, min_score)
    table.insert(partition_stats, max_score)
end

return {
    total_count,
    total_messages_in_queues,
    global_min_score or 0,
    global_max_score or 0,
    partition_stats
}
"""

# 在模块加载时完成替换
REBALANCE_PARTITIONS_SCRIPT = REBALANCE_PARTITIONS_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
JOIN_CONSUMER_SCRIPT = JOIN_CONSUMER_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
EXIT_CONSUMER_SCRIPT = EXIT_CONSUMER_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
CLEANUP_INACTIVE_OWNERS_SCRIPT = CLEANUP_INACTIVE_OWNERS_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
