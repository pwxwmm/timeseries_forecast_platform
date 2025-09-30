<template>
  <div class="dashboard">
    <!-- 欢迎区域 -->
    <div class="welcome-section">
      <div class="welcome-content">
        <div class="welcome-text">
          <h1 class="welcome-title">
            <span class="greeting">欢迎回来，</span>
            <span class="username">{{ currentUser || '用户' }}</span>
          </h1>
          <p class="welcome-subtitle">时间序列预测平台 - 让数据预测更智能</p>
        </div>
        <div class="welcome-actions">
          <el-button type="primary" size="large" @click="$router.push('/tasks/new')">
            <el-icon><Plus /></el-icon>
            创建新任务
          </el-button>
          <el-button size="large" @click="refreshStats" :loading="loading">
            <el-icon><Refresh /></el-icon>
            刷新数据
          </el-button>
        </div>
      </div>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-grid">
      <div class="stat-card tasks" @click="$router.push('/tasks')">
        <div class="stat-background">
          <div class="stat-pattern"></div>
        </div>
        <div class="stat-content">
          <div class="stat-header">
            <div class="stat-icon">
              <el-icon><List /></el-icon>
            </div>
            <div class="stat-trend" :class="getTrendClass('tasks')">
              <el-icon><TrendCharts /></el-icon>
              <span>+12%</span>
            </div>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ stats.tasks?.total || 0 }}</div>
            <div class="stat-label">总任务数</div>
            <div class="stat-detail">
              <span class="running">{{ stats.tasks?.running || 0 }} 运行中</span>
              <span class="completed">{{ stats.tasks?.completed || 0 }} 已完成</span>
            </div>
          </div>
        </div>
      </div>

      <div class="stat-card models" @click="$router.push('/models')">
        <div class="stat-background">
          <div class="stat-pattern"></div>
        </div>
        <div class="stat-content">
          <div class="stat-header">
            <div class="stat-icon">
              <el-icon><Cpu /></el-icon>
            </div>
            <div class="stat-trend" :class="getTrendClass('models')">
              <el-icon><TrendCharts /></el-icon>
              <span>+8%</span>
            </div>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ stats.models?.total || 0 }}</div>
            <div class="stat-label">总模型数</div>
            <div class="stat-detail">
              <span class="training">{{ stats.models?.training || 0 }} 训练中</span>
              <span class="ready">{{ stats.models?.ready || 0 }} 就绪</span>
            </div>
          </div>
        </div>
      </div>

      <div class="stat-card users">
        <div class="stat-background">
          <div class="stat-pattern"></div>
        </div>
        <div class="stat-content">
          <div class="stat-header">
            <div class="stat-icon">
              <el-icon><User /></el-icon>
            </div>
            <div class="stat-trend" :class="getTrendClass('users')">
              <el-icon><TrendCharts /></el-icon>
              <span>+5%</span>
            </div>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ stats.users || 0 }}</div>
            <div class="stat-label">活跃用户</div>
            <div class="stat-detail">
              <span class="online">{{ stats.users || 0 }} 在线</span>
              <span class="total">{{ stats.users || 0 }} 总计</span>
            </div>
          </div>
        </div>
      </div>

      <div class="stat-card performance">
        <div class="stat-background">
          <div class="stat-pattern"></div>
        </div>
        <div class="stat-content">
          <div class="stat-header">
            <div class="stat-icon">
              <el-icon><Loading /></el-icon>
            </div>
            <div class="stat-trend" :class="getTrendClass('performance')">
              <el-icon><TrendCharts /></el-icon>
              <span>+15%</span>
            </div>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ (stats.tasks?.running || 0) + (stats.models?.training || 0) }}</div>
            <div class="stat-label">运行中任务</div>
            <div class="stat-detail">
              <span class="cpu">CPU: 65%</span>
              <span class="memory">内存: 78%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 任务趋势图 -->
    <div class="trend-chart-section">
      <TaskTrendChart />
    </div>

    <!-- 主要内容区域 -->
    <div class="main-content-grid">
      <!-- 最近任务 -->
      <div class="content-card recent-tasks">
        <div class="card-header">
          <div class="header-left">
            <h3 class="card-title">最近任务</h3>
            <p class="card-subtitle">查看最新的预测任务</p>
          </div>
          <el-button type="primary" size="small" @click="$router.push('/tasks')">
            查看全部
            <el-icon><ArrowRight /></el-icon>
          </el-button>
        </div>
        <div class="card-content">
          <div v-if="recentTasks.length === 0" class="empty-state">
            <div class="empty-icon">
              <el-icon><List /></el-icon>
            </div>
            <p class="empty-text">暂无任务</p>
            <el-button type="primary" size="small" @click="$router.push('/tasks/new')">
              创建第一个任务
            </el-button>
          </div>
          <div v-else class="task-list">
            <div
              v-for="task in recentTasks"
              :key="task.id"
              class="task-item"
              @click="$router.push(`/tasks/${task.id}`)"
            >
              <div class="task-avatar">
                <el-avatar :size="40" :style="{ backgroundColor: getTaskColor(task.status) }">
                  <el-icon><TrendCharts /></el-icon>
                </el-avatar>
              </div>
              <div class="task-info">
                <div class="task-name">{{ task.name }}</div>
                <div class="task-meta">
                  <span class="task-user">
                    <el-icon><User /></el-icon>
                    {{ task.user }}
                  </span>
                  <span class="task-time">
                    <el-icon><Clock /></el-icon>
                    {{ utils.formatTime(task.created_at) }}
                  </span>
                </div>
              </div>
              <div class="task-status">
                <el-tag
                  :type="getStatusType(task.status)"
                  size="small"
                  effect="light"
                >
                  {{ utils.getStatusText(task.status) }}
                </el-tag>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 系统状态 -->
      <div class="content-card system-status">
        <div class="card-header">
          <div class="header-left">
            <h3 class="card-title">系统状态</h3>
            <p class="card-subtitle">实时监控系统运行状态</p>
          </div>
          <el-button
            type="text"
            size="small"
            @click="refreshStats"
            :loading="loading"
          >
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
        <div class="card-content">
          <div class="status-grid">
            <div class="status-item">
              <div class="status-icon" :class="{ online: apiStatus, offline: !apiStatus }">
                <el-icon><Connection /></el-icon>
              </div>
              <div class="status-info">
                <div class="status-label">API 服务</div>
                <div class="status-value" :class="{ success: apiStatus, error: !apiStatus }">
                  {{ apiStatus ? '正常运行' : '服务异常' }}
                </div>
              </div>
            </div>
            
            <div class="status-item">
              <div class="status-icon" :class="{ online: prometheusStatus, offline: !prometheusStatus }">
                <el-icon><Monitor /></el-icon>
              </div>
              <div class="status-info">
                <div class="status-label">Prometheus</div>
                <div class="status-value" :class="{ success: prometheusStatus, error: !prometheusStatus }">
                  {{ prometheusStatus ? '连接正常' : '连接失败' }}
                </div>
              </div>
            </div>
            
            <div class="status-item">
              <div class="status-icon online">
                <el-icon><Clock /></el-icon>
              </div>
              <div class="status-info">
                <div class="status-label">最后更新</div>
                <div class="status-value">{{ lastUpdateTime }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快速操作 -->
    <el-row :gutter="20" class="quick-actions">
      <el-col :span="24">
        <el-card class="content-card">
          <template #header>
            <span>快速操作</span>
          </template>
          <div class="action-buttons">
            <el-button
              type="primary"
              size="large"
              @click="$router.push('/tasks/new')"
            >
              <el-icon><Plus /></el-icon>
              创建新任务
            </el-button>
            <el-button
              type="success"
              size="large"
              @click="$router.push('/models')"
            >
              <el-icon><Cpu /></el-icon>
              管理模型
            </el-button>
            <el-button
              type="info"
              size="large"
              @click="showPredictionDialog = true"
            >
              <el-icon><TrendCharts /></el-icon>
              快速预测
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 快速预测对话框 -->
    <el-dialog
      v-model="showPredictionDialog"
      title="快速预测"
      width="500px"
    >
      <el-form
        ref="predictionFormRef"
        :model="predictionForm"
        :rules="predictionRules"
        label-width="100px"
      >
        <el-form-item label="用户" prop="user">
          <el-input
            v-model="predictionForm.user"
            placeholder="请输入用户名"
            clearable
          />
        </el-form-item>
        <el-form-item label="指标查询" prop="metric_query">
          <el-input
            v-model="predictionForm.metric_query"
            placeholder="例如: storage_used_bytes{user='alice'}"
            clearable
          />
        </el-form-item>
        <el-form-item label="预测步数" prop="prediction_steps">
          <el-input-number
            v-model="predictionForm.prediction_steps"
            :min="1"
            :max="24"
            placeholder="预测步数"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showPredictionDialog = false">取消</el-button>
          <el-button
            type="primary"
            @click="handleQuickPrediction"
            :loading="predictionLoading"
          >
            开始预测
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { api, utils } from '../api'
import TaskTrendChart from './TaskTrendChart.vue'

// 响应式数据
const loading = ref(false)
const stats = ref({})
const recentTasks = ref([])
const apiStatus = ref(true)
const prometheusStatus = ref(true)
const lastUpdateTime = ref('')
const showPredictionDialog = ref(false)
const predictionLoading = ref(false)
const predictionFormRef = ref()
const currentUser = ref('')

const predictionForm = reactive({
  user: '',
  metric_query: '',
  prediction_steps: 1
})

const predictionRules = {
  user: [
    { required: true, message: '请输入用户名', trigger: 'blur' }
  ],
  metric_query: [
    { required: true, message: '请输入指标查询', trigger: 'blur' }
  ],
  prediction_steps: [
    { required: true, message: '请输入预测步数', trigger: 'blur' }
  ]
}

// 计算属性
const getStatusType = (status) => {
  const types = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    training: 'warning'
  }
  return types[status] || 'info'
}

const getTaskColor = (status) => {
  const colors = {
    pending: '#409eff',
    running: '#e6a23c',
    completed: '#67c23a',
    failed: '#f56c6c',
    training: '#e6a23c'
  }
  return colors[status] || '#909399'
}

const getTrendClass = (type) => {
  return 'positive' // 可以后续根据实际数据动态计算
}

// 方法
const refreshStats = async () => {
  loading.value = true
  try {
    const [statsData, healthData] = await Promise.all([
      api.getStats(),
      api.healthCheck()
    ])
    
    stats.value = statsData
    apiStatus.value = healthData.status === 'healthy'
    lastUpdateTime.value = new Date().toLocaleString('zh-CN')
    
    // 获取最近任务
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      const tasks = await api.getUserTasks(currentUser)
      recentTasks.value = tasks.slice(0, 5) // 只显示最近5个任务
    }
    
  } catch (error) {
    console.error('Failed to refresh stats:', error)
    apiStatus.value = false
  } finally {
    loading.value = false
  }
}

const handleQuickPrediction = async () => {
  if (!predictionFormRef.value) return
  
  try {
    await predictionFormRef.value.validate()
    predictionLoading.value = true
    
    const result = await api.predict(predictionForm)
    
    ElMessage.success('预测完成')
    console.log('Prediction result:', result)
    
    // 重置表单
    predictionForm.user = ''
    predictionForm.metric_query = ''
    predictionForm.prediction_steps = 1
    showPredictionDialog.value = false
    
  } catch (error) {
    console.error('Prediction failed:', error)
  } finally {
    predictionLoading.value = false
  }
}

// 生命周期
onMounted(() => {
  // 获取当前用户
  currentUser.value = localStorage.getItem('currentUser') || '用户'
  
  refreshStats()
  
  // 设置自动刷新
  setInterval(refreshStats, 30000) // 每30秒刷新一次
})
</script>

<style scoped>
.dashboard {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* 欢迎区域 */
.welcome-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  padding: 40px;
  margin-bottom: 32px;
  color: white;
  position: relative;
  overflow: hidden;
}

.welcome-section::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
  animation: float 6s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  50% { transform: translate(-20px, -20px) rotate(180deg); }
}

.welcome-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: relative;
  z-index: 1;
}

.welcome-text {
  flex: 1;
}

.welcome-title {
  font-size: 36px;
  font-weight: 700;
  margin-bottom: 12px;
  line-height: 1.2;
}

.greeting {
  display: block;
  font-size: 18px;
  opacity: 0.9;
  margin-bottom: 8px;
}

.username {
  background: linear-gradient(45deg, #ff6b6b, #feca57);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.welcome-subtitle {
  font-size: 16px;
  opacity: 0.8;
  margin: 0;
}

.welcome-actions {
  display: flex;
  gap: 16px;
}

.welcome-actions .el-button {
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.welcome-actions .el-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

/* 统计卡片网格 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
  max-width: 100%;
}

.stat-card {
  background: white;
  border-radius: 20px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.stat-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

.stat-background {
  position: absolute;
  top: 0;
  right: 0;
  width: 120px;
  height: 120px;
  opacity: 0.1;
  border-radius: 50%;
}

.stat-card.tasks .stat-background {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.stat-card.models .stat-background {
  background: linear-gradient(135deg, #f093fb, #f5576c);
}

.stat-card.users .stat-background {
  background: linear-gradient(135deg, #4facfe, #00f2fe);
}

.stat-card.performance .stat-background {
  background: linear-gradient(135deg, #43e97b, #38f9d7);
}

.stat-pattern {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 60px;
  height: 60px;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill="currentColor"/></svg>') no-repeat center;
  background-size: contain;
  opacity: 0.3;
}

.stat-content {
  position: relative;
  z-index: 1;
}

.stat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
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

.stat-card.tasks .stat-icon {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.stat-card.models .stat-icon {
  background: linear-gradient(135deg, #f093fb, #f5576c);
}

.stat-card.users .stat-icon {
  background: linear-gradient(135deg, #4facfe, #00f2fe);
}

.stat-card.performance .stat-icon {
  background: linear-gradient(135deg, #43e97b, #38f9d7);
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 8px;
}

.stat-trend.positive {
  background: rgba(34, 197, 94, 0.1);
  color: #16a34a;
}

.stat-body {
  text-align: left;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 12px;
}

.stat-detail {
  display: flex;
  gap: 16px;
  font-size: 12px;
}

.stat-detail span {
  padding: 4px 8px;
  border-radius: 6px;
  font-weight: 500;
}

.stat-detail .running {
  background: rgba(59, 130, 246, 0.1);
  color: #2563eb;
}

.stat-detail .completed {
  background: rgba(34, 197, 94, 0.1);
  color: #16a34a;
}

.stat-detail .training {
  background: rgba(245, 158, 11, 0.1);
  color: #d97706;
}

.stat-detail .ready {
  background: rgba(34, 197, 94, 0.1);
  color: #16a34a;
}

.stat-detail .online {
  background: rgba(34, 197, 94, 0.1);
  color: #16a34a;
}

.stat-detail .total {
  background: rgba(107, 114, 128, 0.1);
  color: #6b7280;
}

.stat-detail .cpu {
  background: rgba(239, 68, 68, 0.1);
  color: #dc2626;
}

.stat-detail .memory {
  background: rgba(245, 158, 11, 0.1);
  color: #d97706;
}

/* 趋势图区域 */
.trend-chart-section {
  margin-bottom: 24px;
}

/* 主要内容网格 */
.main-content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
  max-width: 100%;
}

.content-card {
  background: white;
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.content-card:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
}

.header-left {
  flex: 1;
}

.card-title {
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 4px 0;
}

.card-subtitle {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

.card-content {
  min-height: 200px;
}

/* 任务列表 */
.task-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-radius: 12px;
  background: #f8fafc;
  cursor: pointer;
  transition: all 0.3s ease;
}

.task-item:hover {
  background: #e2e8f0;
  transform: translateX(4px);
}

.task-avatar {
  flex-shrink: 0;
}

.task-info {
  flex: 1;
  min-width: 0;
}

.task-name {
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.task-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #6b7280;
}

.task-meta span {
  display: flex;
  align-items: center;
  gap: 4px;
}

.task-status {
  flex-shrink: 0;
}

/* 系统状态 */
.status-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-radius: 12px;
  background: #f8fafc;
}

.status-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: white;
}

.status-icon.online {
  background: linear-gradient(135deg, #10b981, #059669);
}

.status-icon.offline {
  background: linear-gradient(135deg, #ef4444, #dc2626);
}

.status-info {
  flex: 1;
}

.status-label {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 4px;
}

.status-value {
  font-size: 14px;
  font-weight: 600;
}

.status-value.success {
  color: #16a34a;
}

.status-value.error {
  color: #dc2626;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
}

.empty-icon {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, #e5e7eb, #d1d5db);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: #9ca3af;
  margin-bottom: 16px;
}

.empty-text {
  font-size: 16px;
  color: #6b7280;
  margin-bottom: 16px;
}

/* 快速操作 */
.quick-actions {
  margin-bottom: 32px;
}

.action-buttons {
  display: flex;
  gap: 16px;
  justify-content: center;
  padding: 24px;
  background: white;
  border-radius: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.action-buttons .el-button {
  border-radius: 12px;
  padding: 16px 32px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.action-buttons .el-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* 对话框 */
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

/* 响应式设计 */
@media (min-width: 1600px) {
  .stats-grid {
    grid-template-columns: repeat(4, 1fr);
  }
  
  .main-content-grid {
    grid-template-columns: 2fr 1fr;
  }
}

@media (max-width: 1400px) {
  .stats-grid {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }
}

@media (max-width: 1200px) {
  .main-content-grid {
    grid-template-columns: 1fr;
  }
  
  .stats-grid {
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  }
}

@media (max-width: 768px) {
  .welcome-content {
    flex-direction: column;
    gap: 24px;
    text-align: center;
  }
  
  .welcome-title {
    font-size: 28px;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
  }
}
</style>

