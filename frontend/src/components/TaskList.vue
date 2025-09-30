<template>
  <div class="task-list">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>任务管理</h1>
      <p>管理时间序列预测任务</p>
    </div>

    <!-- 操作栏 -->
    <div class="action-bar">
      <div class="action-left">
        <el-button type="primary" @click="$router.push('/tasks/new')">
          <el-icon><Plus /></el-icon>
          创建任务
        </el-button>
        <el-button @click="refreshTasks" :loading="loading">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>
      <div class="action-right">
        <el-input
          v-model="searchQuery"
          placeholder="搜索任务名称或用户"
          clearable
          style="width: 300px"
          @input="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
      </div>
    </div>

    <!-- 任务表格 -->
    <el-card class="table-card">
      <el-table
        :data="filteredTasks"
        v-loading="loading"
        stripe
        style="width: 100%"
        @sort-change="handleSortChange"
      >
        <el-table-column prop="name" label="任务名称" min-width="200">
          <template #default="{ row }">
            <div class="task-name">
              <span class="name">{{ row.name }}</span>
              <div class="meta">
                <span class="user">{{ row.user }}</span>
                <span class="time">{{ utils.formatTime(row.created_at) }}</span>
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column prop="metric_query" label="指标查询" min-width="250">
          <template #default="{ row }">
            <el-tooltip :content="row.metric_query" placement="top">
              <span class="metric-query">{{ row.metric_query }}</span>
            </el-tooltip>
          </template>
        </el-table-column>

        <el-table-column prop="status" label="状态" width="120" sortable>
          <template #default="{ row }">
            <el-tag
              :type="getStatusType(row.status)"
              size="small"
            >
              {{ utils.getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column prop="config" label="配置" width="150">
          <template #default="{ row }">
            <div class="config-info">
              <div>序列长度: {{ row.config.sequence_length || 24 }}</div>
              <div>预测步数: {{ row.config.prediction_steps || 1 }}</div>
              <div>训练轮数: {{ row.config.epochs || 100 }}</div>
            </div>
          </template>
        </el-table-column>

        <el-table-column prop="updated_at" label="更新时间" width="180" sortable>
          <template #default="{ row }">
            {{ utils.formatTime(row.updated_at) }}
          </template>
        </el-table-column>

        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <div class="action-buttons">
              <el-button
                type="primary"
                size="small"
                @click="viewTask(row)"
              >
                查看
              </el-button>
              <el-button
                type="success"
                size="small"
                @click="predictTask(row)"
                :disabled="row.status !== 'completed'"
              >
                预测
              </el-button>
              <el-button
                type="danger"
                size="small"
                @click="deleteTask(row)"
                :disabled="row.status === 'running'"
              >
                删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="totalTasks"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 任务详情对话框 -->
    <el-dialog
      v-model="showTaskDialog"
      :title="selectedTask?.name || '任务详情'"
      width="800px"
    >
      <div v-if="selectedTask" class="task-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="任务ID">
            {{ selectedTask.id }}
          </el-descriptions-item>
          <el-descriptions-item label="用户">
            {{ selectedTask.user }}
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(selectedTask.status)">
              {{ utils.getStatusText(selectedTask.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ utils.formatTime(selectedTask.created_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="更新时间">
            {{ utils.formatTime(selectedTask.updated_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="模型ID">
            {{ selectedTask.model_id || '未关联' }}
          </el-descriptions-item>
        </el-descriptions>

        <div class="detail-section">
          <h4>指标查询</h4>
          <el-input
            :value="selectedTask.metric_query"
            readonly
            type="textarea"
            :rows="2"
          />
        </div>

        <div class="detail-section">
          <h4>配置参数</h4>
          <el-input
            :value="JSON.stringify(selectedTask.config, null, 2)"
            readonly
            type="textarea"
            :rows="6"
          />
        </div>

        <div v-if="selectedTask.results" class="detail-section">
          <h4>训练结果</h4>
          <el-input
            :value="JSON.stringify(selectedTask.results, null, 2)"
            readonly
            type="textarea"
            :rows="8"
          />
        </div>

        <div v-if="selectedTask.error_message" class="detail-section">
          <h4>错误信息</h4>
          <el-alert
            :title="selectedTask.error_message"
            type="error"
            show-icon
          />
        </div>
      </div>
    </el-dialog>

    <!-- 预测对话框 -->
    <el-dialog
      v-model="showPredictionDialog"
      title="任务预测"
      width="600px"
    >
      <div v-if="selectedTask" class="prediction-form">
        <el-form
          ref="predictionFormRef"
          :model="predictionForm"
          :rules="predictionRules"
          label-width="100px"
        >
          <el-form-item label="用户">
            <el-input :value="selectedTask.user" readonly />
          </el-form-item>
          <el-form-item label="指标查询">
            <el-input :value="selectedTask.metric_query" readonly />
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

        <div v-if="predictionResult" class="prediction-result">
          <h4>预测结果</h4>
          <div class="result-item">
            <span class="label">预测值:</span>
            <span class="value">{{ predictionResult.predictions.join(', ') }}</span>
          </div>
          <div class="result-item">
            <span class="label">时间戳:</span>
            <span class="value">{{ predictionResult.timestamps.join(', ') }}</span>
          </div>
        </div>
      </div>

      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showPredictionDialog = false">关闭</el-button>
          <el-button
            type="primary"
            @click="handlePrediction"
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
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api, utils } from '../api'

// 响应式数据
const loading = ref(false)
const tasks = ref([])
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(20)
const totalTasks = ref(0)
const showTaskDialog = ref(false)
const showPredictionDialog = ref(false)
const selectedTask = ref(null)
const predictionLoading = ref(false)
const predictionResult = ref(null)
const predictionFormRef = ref()

const predictionForm = reactive({
  prediction_steps: 1
})

const predictionRules = {
  prediction_steps: [
    { required: true, message: '请输入预测步数', trigger: 'blur' }
  ]
}

// 计算属性
const filteredTasks = computed(() => {
  let filtered = tasks.value

  // 搜索过滤
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(task =>
      task.name.toLowerCase().includes(query) ||
      task.user.toLowerCase().includes(query)
    )
  }

  // 分页
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filtered.slice(start, end)
})

// 方法
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

const loadTasks = async () => {
  loading.value = true
  try {
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      const userTasks = await api.getUserTasks(currentUser)
      tasks.value = userTasks
      totalTasks.value = userTasks.length
    }
  } catch (error) {
    console.error('Failed to load tasks:', error)
    ElMessage.error('加载任务失败')
  } finally {
    loading.value = false
  }
}

const refreshTasks = () => {
  loadTasks()
}

const handleSearch = () => {
  currentPage.value = 1
}

const handleSortChange = ({ prop, order }) => {
  if (order === 'ascending') {
    tasks.value.sort((a, b) => a[prop] > b[prop] ? 1 : -1)
  } else if (order === 'descending') {
    tasks.value.sort((a, b) => a[prop] < b[prop] ? 1 : -1)
  }
}

const handleSizeChange = (size) => {
  pageSize.value = size
  currentPage.value = 1
}

const handleCurrentChange = (page) => {
  currentPage.value = page
}

const viewTask = (task) => {
  selectedTask.value = task
  showTaskDialog.value = true
}

const predictTask = (task) => {
  selectedTask.value = task
  predictionForm.prediction_steps = 1
  predictionResult.value = null
  showPredictionDialog.value = true
}

const handlePrediction = async () => {
  if (!predictionFormRef.value) return

  try {
    await predictionFormRef.value.validate()
    predictionLoading.value = true

    const predictionData = {
      user: selectedTask.value.user,
      metric_query: selectedTask.value.metric_query,
      prediction_steps: predictionForm.prediction_steps
    }

    const result = await api.predict(predictionData)
    predictionResult.value = result

    ElMessage.success('预测完成')

  } catch (error) {
    console.error('Prediction failed:', error)
  } finally {
    predictionLoading.value = false
  }
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
    loadTasks()

  } catch (error) {
    if (error !== 'cancel') {
      console.error('Delete task failed:', error)
      ElMessage.error('删除任务失败')
    }
  }
}

// 生命周期
onMounted(() => {
  loadTasks()
})
</script>

<style scoped>
.task-list {
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

.action-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.action-left {
  display: flex;
  gap: 12px;
}

.table-card {
  margin-bottom: 20px;
}

.task-name .name {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  display: block;
  margin-bottom: 4px;
}

.task-name .meta {
  font-size: 12px;
  color: #909399;
}

.task-name .user {
  margin-right: 12px;
}

.metric-query {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #606266;
  display: block;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.config-info {
  font-size: 12px;
  color: #606266;
  line-height: 1.4;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.pagination {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}

.task-detail {
  max-height: 600px;
  overflow-y: auto;
}

.detail-section {
  margin-top: 20px;
}

.detail-section h4 {
  font-size: 16px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 12px;
}

.prediction-form {
  max-height: 500px;
  overflow-y: auto;
}

.prediction-result {
  margin-top: 20px;
  padding: 16px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.prediction-result h4 {
  font-size: 16px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 12px;
}

.result-item {
  display: flex;
  margin-bottom: 8px;
}

.result-item .label {
  font-weight: 500;
  color: #606266;
  margin-right: 8px;
  min-width: 80px;
}

.result-item .value {
  color: #303133;
  font-family: 'Courier New', monospace;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>
