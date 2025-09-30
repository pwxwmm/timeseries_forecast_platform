<template>
  <div class="model-manager">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>模型管理</h1>
      <p>管理时间序列预测模型</p>
    </div>

    <!-- 操作栏 -->
    <div class="action-bar">
      <div class="action-left">
        <el-button @click="refreshModels" :loading="loading">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>
      <div class="action-right">
        <el-select
          v-model="statusFilter"
          placeholder="状态筛选"
          clearable
          style="width: 150px; margin-right: 12px"
        >
          <el-option label="全部" value="" />
          <el-option label="训练中" value="training" />
          <el-option label="已完成" value="completed" />
          <el-option label="失败" value="failed" />
        </el-select>
        <el-input
          v-model="searchQuery"
          placeholder="搜索模型名称"
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

    <!-- 模型表格 -->
    <el-card class="table-card">
      <el-table
        :data="filteredModels"
        v-loading="loading"
        stripe
        style="width: 100%"
        @sort-change="handleSortChange"
      >
        <el-table-column prop="name" label="模型名称" min-width="200">
          <template #default="{ row }">
            <div class="model-name">
              <span class="name">{{ row.name }}</span>
              <div class="meta">
                <span class="user">{{ row.user }}</span>
                <span class="time">{{ utils.formatTime(row.created_at) }}</span>
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column prop="model_type" label="模型类型" width="120">
          <template #default="{ row }">
            <el-tag type="primary" size="small">
              {{ row.model_type.toUpperCase() }}
            </el-tag>
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

        <el-table-column prop="config" label="配置" width="200">
          <template #default="{ row }">
            <div class="config-info">
              <div>隐藏层: {{ row.config.hidden_dim || 64 }}</div>
              <div>层数: {{ row.config.num_layers || 2 }}</div>
              <div>序列长度: {{ row.config.sequence_length || 24 }}</div>
            </div>
          </template>
        </el-table-column>

        <el-table-column prop="metrics" label="性能指标" width="200">
          <template #default="{ row }">
            <div v-if="row.metrics" class="metrics-info">
              <div>MSE: {{ row.metrics.mse?.toFixed(6) || 'N/A' }}</div>
              <div>MAE: {{ row.metrics.mae?.toFixed(4) || 'N/A' }}</div>
              <div>RMSE: {{ row.metrics.rmse?.toFixed(4) || 'N/A' }}</div>
              <div>MAPE: {{ row.metrics.mape?.toFixed(2) || 'N/A' }}%</div>
            </div>
            <div v-else class="no-metrics">
              <el-tag type="info" size="small">暂无指标</el-tag>
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
                @click="viewModel(row)"
              >
                查看
              </el-button>
              <el-button
                type="success"
                size="small"
                @click="downloadModel(row)"
                :disabled="row.status !== 'completed'"
              >
                下载
              </el-button>
              <el-button
                type="danger"
                size="small"
                @click="deleteModel(row)"
                :disabled="row.status === 'training'"
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
          :total="totalModels"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 模型详情对话框 -->
    <el-dialog
      v-model="showModelDialog"
      :title="selectedModel?.name || '模型详情'"
      width="800px"
    >
      <div v-if="selectedModel" class="model-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="模型ID">
            {{ selectedModel.id }}
          </el-descriptions-item>
          <el-descriptions-item label="用户">
            {{ selectedModel.user }}
          </el-descriptions-item>
          <el-descriptions-item label="任务ID">
            {{ selectedModel.task_id }}
          </el-descriptions-item>
          <el-descriptions-item label="模型类型">
            <el-tag type="primary">{{ selectedModel.model_type.toUpperCase() }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(selectedModel.status)">
              {{ utils.getStatusText(selectedModel.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ utils.formatTime(selectedModel.created_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="更新时间">
            {{ utils.formatTime(selectedModel.updated_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="文件路径">
            {{ selectedModel.file_path || '未保存' }}
          </el-descriptions-item>
        </el-descriptions>

        <div class="detail-section">
          <h4>模型配置</h4>
          <el-input
            :value="JSON.stringify(selectedModel.config, null, 2)"
            readonly
            type="textarea"
            :rows="8"
          />
        </div>

        <div v-if="selectedModel.metrics" class="detail-section">
          <h4>性能指标</h4>
          <el-row :gutter="20">
            <el-col :span="6">
              <div class="metric-card">
                <div class="metric-value">{{ selectedModel.metrics.mse?.toFixed(6) || 'N/A' }}</div>
                <div class="metric-label">MSE</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="metric-card">
                <div class="metric-value">{{ selectedModel.metrics.mae?.toFixed(4) || 'N/A' }}</div>
                <div class="metric-label">MAE</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="metric-card">
                <div class="metric-value">{{ selectedModel.metrics.rmse?.toFixed(4) || 'N/A' }}</div>
                <div class="metric-label">RMSE</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="metric-card">
                <div class="metric-value">{{ selectedModel.metrics.mape?.toFixed(2) || 'N/A' }}%</div>
                <div class="metric-label">MAPE</div>
              </div>
            </el-col>
          </el-row>
        </div>

        <div v-if="selectedModel.error_message" class="detail-section">
          <h4>错误信息</h4>
          <el-alert
            :title="selectedModel.error_message"
            type="error"
            show-icon
          />
        </div>
      </div>
    </el-dialog>

    <!-- 模型性能图表 -->
    <el-dialog
      v-model="showMetricsDialog"
      title="模型性能分析"
      width="1000px"
    >
      <div v-if="selectedModel && selectedModel.metrics" class="metrics-chart">
        <div class="chart-container">
          <h4>性能指标对比</h4>
          <div class="metrics-comparison">
            <div class="metric-item">
              <span class="label">均方误差 (MSE):</span>
              <span class="value">{{ selectedModel.metrics.mse?.toFixed(6) || 'N/A' }}</span>
            </div>
            <div class="metric-item">
              <span class="label">平均绝对误差 (MAE):</span>
              <span class="value">{{ selectedModel.metrics.mae?.toFixed(4) || 'N/A' }}</span>
            </div>
            <div class="metric-item">
              <span class="label">均方根误差 (RMSE):</span>
              <span class="value">{{ selectedModel.metrics.rmse?.toFixed(4) || 'N/A' }}</span>
            </div>
            <div class="metric-item">
              <span class="label">平均绝对百分比误差 (MAPE):</span>
              <span class="value">{{ selectedModel.metrics.mape?.toFixed(2) || 'N/A' }}%</span>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api, utils } from '../api'

// 响应式数据
const loading = ref(false)
const models = ref([])
const searchQuery = ref('')
const statusFilter = ref('')
const currentPage = ref(1)
const pageSize = ref(20)
const totalModels = ref(0)
const showModelDialog = ref(false)
const showMetricsDialog = ref(false)
const selectedModel = ref(null)

// 计算属性
const filteredModels = computed(() => {
  let filtered = models.value

  // 状态过滤
  if (statusFilter.value) {
    filtered = filtered.filter(model => model.status === statusFilter.value)
  }

  // 搜索过滤
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(model =>
      model.name.toLowerCase().includes(query) ||
      model.user.toLowerCase().includes(query)
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
    training: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

const loadModels = async () => {
  loading.value = true
  try {
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      const userModels = await api.getUserModels(currentUser)
      models.value = userModels
      totalModels.value = userModels.length
    }
  } catch (error) {
    console.error('Failed to load models:', error)
    ElMessage.error('加载模型失败')
  } finally {
    loading.value = false
  }
}

const refreshModels = () => {
  loadModels()
}

const handleSearch = () => {
  currentPage.value = 1
}

const handleSortChange = ({ prop, order }) => {
  if (order === 'ascending') {
    models.value.sort((a, b) => a[prop] > b[prop] ? 1 : -1)
  } else if (order === 'descending') {
    models.value.sort((a, b) => a[prop] < b[prop] ? 1 : -1)
  }
}

const handleSizeChange = (size) => {
  pageSize.value = size
  currentPage.value = 1
}

const handleCurrentChange = (page) => {
  currentPage.value = page
}

const viewModel = (model) => {
  selectedModel.value = model
  showModelDialog.value = true
}

const downloadModel = async (model) => {
  if (!model.file_path) {
    ElMessage.warning('模型文件不存在')
    return
  }

  try {
    // 这里应该实现模型文件下载逻辑
    ElMessage.success('模型下载功能待实现')
  } catch (error) {
    console.error('Download failed:', error)
    ElMessage.error('下载模型失败')
  }
}

const deleteModel = async (model) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除模型 "${model.name}" 吗？`,
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    await api.deleteModel(model.id)
    ElMessage.success('模型删除成功')
    loadModels()

  } catch (error) {
    if (error !== 'cancel') {
      console.error('Delete model failed:', error)
      ElMessage.error('删除模型失败')
    }
  }
}

// 生命周期
onMounted(() => {
  loadModels()
})
</script>

<style scoped>
.model-manager {
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

.model-name .name {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  display: block;
  margin-bottom: 4px;
}

.model-name .meta {
  font-size: 12px;
  color: #909399;
}

.model-name .user {
  margin-right: 12px;
}

.config-info {
  font-size: 12px;
  color: #606266;
  line-height: 1.4;
}

.metrics-info {
  font-size: 12px;
  color: #606266;
  line-height: 1.4;
}

.no-metrics {
  text-align: center;
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

.model-detail {
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

.metric-card {
  text-align: center;
  padding: 16px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.metric-value {
  font-size: 24px;
  font-weight: 600;
  color: #409eff;
  margin-bottom: 8px;
}

.metric-label {
  font-size: 12px;
  color: #909399;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metrics-chart {
  padding: 20px 0;
}

.chart-container h4 {
  font-size: 18px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 20px;
  text-align: center;
}

.metrics-comparison {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.metric-item .label {
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

.metric-item .value {
  font-size: 16px;
  color: #303133;
  font-weight: 600;
  font-family: 'Courier New', monospace;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .model-manager {
    padding: 0 16px;
  }
  
  .action-bar {
    flex-direction: column;
    gap: 12px;
    align-items: stretch;
  }
  
  .action-right {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .metrics-comparison {
    grid-template-columns: 1fr;
  }
}
</style>
