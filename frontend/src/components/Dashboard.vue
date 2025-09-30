<template>
  <div class="dashboard">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>仪表板</h1>
      <p>时间序列预测平台概览</p>
    </div>

    <!-- 统计卡片 -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon tasks">
              <el-icon><List /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.tasks?.total || 0 }}</div>
              <div class="stat-label">总任务数</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon models">
              <el-icon><Cpu /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.models?.total || 0 }}</div>
              <div class="stat-label">总模型数</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon users">
              <el-icon><User /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.users || 0 }}</div>
              <div class="stat-label">用户数</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon running">
              <el-icon><Loading /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ (stats.tasks?.running || 0) + (stats.models?.training || 0) }}</div>
              <div class="stat-label">运行中</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 主要内容区域 -->
    <el-row :gutter="20" class="main-content">
      <!-- 左侧：最近任务 -->
      <el-col :span="12">
        <el-card class="content-card">
          <template #header>
            <div class="card-header">
              <span>最近任务</span>
              <el-button type="primary" size="small" @click="$router.push('/tasks')">
                查看全部
              </el-button>
            </div>
          </template>
          <div class="task-list">
            <div v-if="recentTasks.length === 0" class="empty-state">
              <el-empty description="暂无任务" />
            </div>
            <div v-else>
              <div
                v-for="task in recentTasks"
                :key="task.id"
                class="task-item"
              >
                <div class="task-info">
                  <div class="task-name">{{ task.name }}</div>
                  <div class="task-meta">
                    <span class="task-user">{{ task.user }}</span>
                    <span class="task-time">{{ utils.formatTime(task.created_at) }}</span>
                  </div>
                </div>
                <div class="task-status">
                  <el-tag
                    :type="getStatusType(task.status)"
                    size="small"
                  >
                    {{ utils.getStatusText(task.status) }}
                  </el-tag>
                </div>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 右侧：系统状态 -->
      <el-col :span="12">
        <el-card class="content-card">
          <template #header>
            <div class="card-header">
              <span>系统状态</span>
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
          </template>
          <div class="system-status">
            <div class="status-item">
              <div class="status-label">API 服务</div>
              <div class="status-value">
                <el-tag :type="apiStatus ? 'success' : 'danger'" size="small">
                  {{ apiStatus ? '正常' : '异常' }}
                </el-tag>
              </div>
            </div>
            <div class="status-item">
              <div class="status-label">Prometheus</div>
              <div class="status-value">
                <el-tag :type="prometheusStatus ? 'success' : 'danger'" size="small">
                  {{ prometheusStatus ? '连接正常' : '连接失败' }}
                </el-tag>
              </div>
            </div>
            <div class="status-item">
              <div class="status-label">最后更新</div>
              <div class="status-value">
                {{ lastUpdateTime }}
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

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
  refreshStats()
  
  // 设置自动刷新
  setInterval(refreshStats, 30000) // 每30秒刷新一次
})
</script>

<style scoped>
.dashboard {
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 8px;
}

.page-header p {
  color: #909399;
  font-size: 14px;
}

.stats-row {
  margin-bottom: 24px;
}

.stat-card {
  height: 100px;
}

.stat-content {
  display: flex;
  align-items: center;
  height: 100%;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
  font-size: 24px;
  color: white;
}

.stat-icon.tasks {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stat-icon.models {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.stat-icon.users {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.stat-icon.running {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #909399;
}

.main-content {
  margin-bottom: 24px;
}

.content-card {
  height: 400px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-list {
  height: 320px;
  overflow-y: auto;
}

.task-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.task-item:last-child {
  border-bottom: none;
}

.task-info {
  flex: 1;
}

.task-name {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 4px;
}

.task-meta {
  font-size: 12px;
  color: #909399;
}

.task-user {
  margin-right: 12px;
}

.system-status {
  padding: 20px 0;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.status-item:last-child {
  border-bottom: none;
}

.status-label {
  font-size: 14px;
  color: #606266;
}

.status-value {
  font-size: 14px;
  color: #303133;
}

.quick-actions {
  margin-bottom: 24px;
}

.action-buttons {
  display: flex;
  gap: 16px;
  justify-content: center;
  padding: 20px 0;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}
</style>
