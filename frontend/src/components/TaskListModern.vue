<template>
  <div class="task-list-modern">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <div class="header-left">
          <h1 class="page-title">任务管理</h1>
          <p class="page-subtitle">管理时间序列预测任务，监控训练进度</p>
        </div>
        <div class="header-actions">
          <el-button type="primary" size="large" @click="$router.push('/tasks/new')">
            <el-icon><Plus /></el-icon>
            创建新任务
          </el-button>
          <el-button size="large" @click="refreshTasks" :loading="loading">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>
    </div>

    <!-- 统计概览 -->
    <div class="stats-overview">
      <div class="stat-item">
        <div class="stat-icon total">
          <el-icon><List /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ tasks.length }}</div>
          <div class="stat-label">总任务数</div>
        </div>
      </div>
      <div class="stat-item">
        <div class="stat-icon running">
          <el-icon><Loading /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ runningTasks }}</div>
          <div class="stat-label">运行中</div>
        </div>
      </div>
      <div class="stat-item">
        <div class="stat-icon completed">
          <el-icon><Check /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ completedTasks }}</div>
          <div class="stat-label">已完成</div>
        </div>
      </div>
      <div class="stat-item">
        <div class="stat-icon failed">
          <el-icon><Close /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ failedTasks }}</div>
          <div class="stat-label">失败</div>
        </div>
      </div>
    </div>

    <!-- 筛选和搜索 -->
    <div class="filter-section">
      <div class="filter-left">
        <el-input
          v-model="searchQuery"
          placeholder="搜索任务名称、用户或描述..."
          clearable
          class="search-input"
          @input="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        
        <el-select v-model="statusFilter" placeholder="状态筛选" clearable class="filter-select">
          <el-option label="全部状态" value="" />
          <el-option label="待处理" value="pending" />
          <el-option label="运行中" value="running" />
          <el-option label="已完成" value="completed" />
          <el-option label="失败" value="failed" />
        </el-select>
        
        <el-select v-model="userFilter" placeholder="用户筛选" clearable class="filter-select">
          <el-option label="全部用户" value="" />
          <el-option
            v-for="user in uniqueUsers"
            :key="user"
            :label="user"
            :value="user"
          />
        </el-select>
      </div>
      
      <div class="filter-right">
        <el-button-group>
          <el-button
            :type="viewMode === 'grid' ? 'primary' : ''"
            @click="viewMode = 'grid'"
          >
            <el-icon><Grid /></el-icon>
          </el-button>
          <el-button
            :type="viewMode === 'list' ? 'primary' : ''"
            @click="viewMode = 'list'"
          >
            <el-icon><List /></el-icon>
          </el-button>
        </el-button-group>
      </div>
    </div>

    <!-- 任务列表 -->
    <div class="tasks-container">
      <!-- 网格视图 -->
      <div v-if="viewMode === 'grid'" class="tasks-grid">
        <div
          v-for="task in filteredTasks"
          :key="task.id"
          class="task-card"
          @click="viewTask(task)"
        >
          <div class="task-header">
            <div class="task-avatar">
              <el-avatar :size="40" :style="{ backgroundColor: getTaskColor(task.status) }">
                <el-icon><TrendCharts /></el-icon>
              </el-avatar>
            </div>
            <div class="task-status">
              <el-tag
                :type="getStatusType(task.status)"
                size="small"
                effect="light"
              >
                {{ getStatusText(task.status) }}
              </el-tag>
            </div>
          </div>
          
          <div class="task-body">
            <h3 class="task-name">{{ task.name }}</h3>
            <p class="task-description">{{ task.description || '暂无描述' }}</p>
            
            <div class="task-meta">
              <div class="meta-item">
                <el-icon><User /></el-icon>
                <span>{{ task.user }}</span>
              </div>
              <div class="meta-item">
                <el-icon><Clock /></el-icon>
                <span>{{ formatTime(task.created_at) }}</span>
              </div>
            </div>
          </div>
          
          <div class="task-footer">
            <div class="task-progress" v-if="task.status === 'running'">
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: '65%' }"></div>
              </div>
              <span class="progress-text">65%</span>
            </div>
            
            <div class="task-actions">
              <el-button size="small" @click.stop="editTask(task)">
                <el-icon><Edit /></el-icon>
              </el-button>
              <el-button size="small" type="danger" @click.stop="deleteTask(task)">
                <el-icon><Delete /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>

      <!-- 列表视图 -->
      <div v-else class="tasks-list">
        <div class="list-header">
          <div class="header-cell name">任务名称</div>
          <div class="header-cell user">用户</div>
          <div class="header-cell status">状态</div>
          <div class="header-cell created">创建时间</div>
          <div class="header-cell actions">操作</div>
        </div>
        
        <div class="list-body">
          <div
            v-for="task in filteredTasks"
            :key="task.id"
            class="list-item"
            @click="viewTask(task)"
          >
            <div class="list-cell name">
              <div class="task-info">
                <div class="task-avatar-small">
                  <el-avatar :size="32" :style="{ backgroundColor: getTaskColor(task.status) }">
                    <el-icon><TrendCharts /></el-icon>
                  </el-avatar>
                </div>
                <div class="task-details">
                  <div class="task-name">{{ task.name }}</div>
                  <div class="task-description">{{ task.description || '暂无描述' }}</div>
                </div>
              </div>
            </div>
            
            <div class="list-cell user">
              <div class="user-info">
                <el-avatar :size="24">
                  <el-icon><User /></el-icon>
                </el-avatar>
                <span>{{ task.user }}</span>
              </div>
            </div>
            
            <div class="list-cell status">
              <el-tag
                :type="getStatusType(task.status)"
                size="small"
                effect="light"
              >
                {{ getStatusText(task.status) }}
              </el-tag>
            </div>
            
            <div class="list-cell created">
              {{ formatTime(task.created_at) }}
            </div>
            
            <div class="list-cell actions">
              <div class="action-buttons">
                <el-button size="small" @click.stop="editTask(task)">
                  <el-icon><Edit /></el-icon>
                </el-button>
                <el-button size="small" type="danger" @click.stop="deleteTask(task)">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-if="filteredTasks.length === 0" class="empty-state">
      <div class="empty-icon">
        <el-icon><List /></el-icon>
      </div>
      <h3 class="empty-title">暂无任务</h3>
      <p class="empty-description">开始创建您的第一个时间序列预测任务</p>
      <el-button type="primary" size="large" @click="$router.push('/tasks/new')">
        <el-icon><Plus /></el-icon>
        创建新任务
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api, utils } from '../api'

// 响应式数据
const loading = ref(false)
const tasks = ref([])
const searchQuery = ref('')
const statusFilter = ref('')
const userFilter = ref('')
const viewMode = ref('grid')

// 计算属性
const filteredTasks = computed(() => {
  let filtered = tasks.value

  // 搜索过滤
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(task =>
      task.name.toLowerCase().includes(query) ||
      task.user.toLowerCase().includes(query) ||
      (task.description && task.description.toLowerCase().includes(query))
    )
  }

  // 状态过滤
  if (statusFilter.value) {
    filtered = filtered.filter(task => task.status === statusFilter.value)
  }

  // 用户过滤
  if (userFilter.value) {
    filtered = filtered.filter(task => task.user === userFilter.value)
  }

  return filtered
})

const uniqueUsers = computed(() => {
  return [...new Set(tasks.value.map(task => task.user))]
})

const runningTasks = computed(() => tasks.value.filter(task => task.status === 'running').length)
const completedTasks = computed(() => tasks.value.filter(task => task.status === 'completed').length)
const failedTasks = computed(() => tasks.value.filter(task => task.status === 'failed').length)

// 方法
const getStatusType = (status) => {
  const types = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

const getStatusText = (status) => {
  const texts = {
    pending: '待处理',
    running: '运行中',
    completed: '已完成',
    failed: '失败'
  }
  return texts[status] || '未知'
}

const getTaskColor = (status) => {
  const colors = {
    pending: '#409eff',
    running: '#e6a23c',
    completed: '#67c23a',
    failed: '#f56c6c'
  }
  return colors[status] || '#909399'
}

const formatTime = (time) => {
  return utils.formatTime(time)
}

const refreshTasks = async () => {
  loading.value = true
  try {
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      tasks.value = await api.getUserTasks(currentUser)
    }
  } catch (error) {
    console.error('Failed to fetch tasks:', error)
    ElMessage.error('获取任务列表失败')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  // 搜索逻辑已在计算属性中处理
}

const viewTask = (task) => {
  // 跳转到任务详情页面
  console.log('View task:', task)
}

const editTask = (task) => {
  // 跳转到编辑页面
  console.log('Edit task:', task)
}

const deleteTask = async (task) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除任务 "${task.name}" 吗？`,
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await api.deleteTask(task.id)
    ElMessage.success('任务删除成功')
    refreshTasks()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete task:', error)
      ElMessage.error('删除任务失败')
    }
  }
}

// 生命周期
onMounted(() => {
  refreshTasks()
})
</script>

<style scoped>
.task-list-modern {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* 页面头部 */
.page-header {
  background: white;
  border-radius: 20px;
  padding: 32px;
  margin-bottom: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  flex: 1;
}

.page-title {
  font-size: 32px;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.page-subtitle {
  font-size: 16px;
  color: #6b7280;
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.header-actions .el-button {
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 600;
}

/* 统计概览 */
.stats-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 24px;
}

.stat-item {
  background: white;
  border-radius: 16px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.stat-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: white;
}

.stat-icon.total {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.stat-icon.running {
  background: linear-gradient(135deg, #e6a23c, #f39c12);
}

.stat-icon.completed {
  background: linear-gradient(135deg, #67c23a, #27ae60);
}

.stat-icon.failed {
  background: linear-gradient(135deg, #f56c6c, #e74c3c);
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
}

/* 筛选区域 */
.filter-section {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.filter-left {
  display: flex;
  gap: 16px;
  align-items: center;
}

.search-input {
  width: 300px;
}

.filter-select {
  width: 150px;
}

/* 任务容器 */
.tasks-container {
  background: white;
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

/* 网格视图 */
.tasks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

.task-card {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #fafafa;
}

.task-card:hover {
  border-color: #3b82f6;
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.task-body {
  margin-bottom: 16px;
}

.task-name {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.task-description {
  font-size: 14px;
  color: #6b7280;
  margin: 0 0 12px 0;
  line-height: 1.5;
}

.task-meta {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #6b7280;
}

.task-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-progress {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.progress-bar {
  flex: 1;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #1d4ed8);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 12px;
  color: #6b7280;
  font-weight: 600;
}

.task-actions {
  display: flex;
  gap: 8px;
}

/* 列表视图 */
.tasks-list {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
}

.list-header {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 120px;
  gap: 16px;
  padding: 16px 20px;
  background: #f8fafc;
  border-bottom: 1px solid #e5e7eb;
  font-weight: 600;
  color: #374151;
  font-size: 14px;
}

.list-body {
  max-height: 600px;
  overflow-y: auto;
}

.list-item {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 120px;
  gap: 16px;
  padding: 16px 20px;
  border-bottom: 1px solid #f3f4f6;
  cursor: pointer;
  transition: all 0.2s ease;
}

.list-item:hover {
  background: #f8fafc;
}

.list-item:last-child {
  border-bottom: none;
}

.list-cell {
  display: flex;
  align-items: center;
}

.task-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.task-avatar-small {
  flex-shrink: 0;
}

.task-details {
  flex: 1;
  min-width: 0;
}

.task-details .task-name {
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 4px 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.task-details .task-description {
  font-size: 12px;
  color: #6b7280;
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 80px 20px;
}

.empty-icon {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #e5e7eb, #d1d5db);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  color: #9ca3af;
  margin: 0 auto 24px;
}

.empty-title {
  font-size: 24px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.empty-description {
  font-size: 16px;
  color: #6b7280;
  margin: 0 0 24px 0;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .tasks-grid {
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    gap: 20px;
    align-items: stretch;
  }
  
  .stats-overview {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .filter-section {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }
  
  .filter-left {
    flex-direction: column;
    align-items: stretch;
  }
  
  .search-input,
  .filter-select {
    width: 100%;
  }
  
  .tasks-grid {
    grid-template-columns: 1fr;
  }
  
  .list-header,
  .list-item {
    grid-template-columns: 1fr;
    gap: 8px;
  }
  
  .list-cell {
    justify-content: flex-start;
  }
}
</style>
