<template>
  <div class="model-manager-modern">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <div class="header-left">
          <h1 class="page-title">模型管理</h1>
          <p class="page-subtitle">管理LSTM模型，监控训练状态和性能</p>
        </div>
        <div class="header-actions">
          <el-button type="primary" size="large" @click="showCreateDialog = true">
            <el-icon><Plus /></el-icon>
            创建新模型
          </el-button>
          <el-button size="large" @click="refreshModels" :loading="loading">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>
    </div>

    <!-- 模型统计 -->
    <div class="models-stats">
      <div class="stat-card">
        <div class="stat-icon total">
          <el-icon><Cpu /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ models.length }}</div>
          <div class="stat-label">总模型数</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon training">
          <el-icon><Loading /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ trainingModels }}</div>
          <div class="stat-label">训练中</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon ready">
          <el-icon><Check /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ readyModels }}</div>
          <div class="stat-label">就绪</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon failed">
          <el-icon><Close /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ failedModels }}</div>
          <div class="stat-label">失败</div>
        </div>
      </div>
    </div>

    <!-- 模型列表 -->
    <div class="models-container">
      <div class="models-grid">
        <div
          v-for="model in models"
          :key="model.id"
          class="model-card"
          :class="{ [model.status]: true }"
        >
          <div class="model-header">
            <div class="model-avatar">
              <el-avatar :size="48" :style="{ backgroundColor: getModelColor(model.status) }">
                <el-icon><Cpu /></el-icon>
              </el-avatar>
            </div>
            <div class="model-status">
              <el-tag
                :type="getStatusType(model.status)"
                size="small"
                effect="light"
              >
                {{ getStatusText(model.status) }}
              </el-tag>
            </div>
          </div>
          
          <div class="model-body">
            <h3 class="model-name">{{ model.name }}</h3>
            <p class="model-description">{{ model.description || '暂无描述' }}</p>
            
            <div class="model-info">
              <div class="info-item">
                <span class="info-label">类型:</span>
                <span class="info-value">{{ model.model_type || 'LSTM' }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">用户:</span>
                <span class="info-value">{{ model.user }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">创建时间:</span>
                <span class="info-value">{{ formatTime(model.created_at) }}</span>
              </div>
            </div>
          </div>
          
          <div class="model-footer">
            <div class="model-progress" v-if="model.status === 'training'">
              <div class="progress-info">
                <span class="progress-label">训练进度</span>
                <span class="progress-percent">{{ model.progress || 0 }}%</span>
              </div>
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: `${model.progress || 0}%` }"></div>
              </div>
            </div>
            
            <div class="model-actions">
              <el-button size="small" @click="viewModel(model)">
                <el-icon><View /></el-icon>
              </el-button>
              <el-button size="small" @click="editModel(model)">
                <el-icon><Edit /></el-icon>
              </el-button>
              <el-button size="small" type="danger" @click="deleteModel(model)">
                <el-icon><Delete /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-if="models.length === 0" class="empty-state">
      <div class="empty-icon">
        <el-icon><Cpu /></el-icon>
      </div>
      <h3 class="empty-title">暂无模型</h3>
      <p class="empty-description">开始创建您的第一个LSTM模型</p>
      <el-button type="primary" size="large" @click="showCreateDialog = true">
        <el-icon><Plus /></el-icon>
        创建新模型
      </el-button>
    </div>

    <!-- 创建模型对话框 -->
    <el-dialog
      v-model="showCreateDialog"
      title="创建新模型"
      width="600px"
      :before-close="handleDialogClose"
    >
      <el-form
        ref="createFormRef"
        :model="createForm"
        :rules="createRules"
        label-width="120px"
      >
        <el-form-item label="模型名称" prop="name">
          <el-input
            v-model="createForm.name"
            placeholder="请输入模型名称"
            clearable
          />
        </el-form-item>
        
        <el-form-item label="模型描述" prop="description">
          <el-input
            v-model="createForm.description"
            type="textarea"
            :rows="3"
            placeholder="请输入模型描述"
            maxlength="500"
            show-word-limit
          />
        </el-form-item>
        
        <el-form-item label="模型类型" prop="model_type">
          <el-select v-model="createForm.model_type" placeholder="选择模型类型">
            <el-option label="LSTM" value="lstm" />
            <el-option label="GRU" value="gru" />
            <el-option label="Transformer" value="transformer" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="隐藏层维度" prop="hidden_dim">
          <el-input-number
            v-model="createForm.hidden_dim"
            :min="32"
            :max="512"
            controls-position="right"
          />
        </el-form-item>
        
        <el-form-item label="层数" prop="num_layers">
          <el-input-number
            v-model="createForm.num_layers"
            :min="1"
            :max="6"
            controls-position="right"
          />
        </el-form-item>
      </el-form>
      
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showCreateDialog = false">取消</el-button>
          <el-button
            type="primary"
            @click="createModel"
            :loading="creating"
          >
            创建模型
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
const creating = ref(false)
const models = ref([])
const showCreateDialog = ref(false)
const createFormRef = ref()

const createForm = reactive({
  name: '',
  description: '',
  model_type: 'lstm',
  hidden_dim: 64,
  num_layers: 2
})

const createRules = {
  name: [
    { required: true, message: '请输入模型名称', trigger: 'blur' },
    { min: 2, max: 50, message: '模型名称长度在 2 到 50 个字符', trigger: 'blur' }
  ],
  model_type: [
    { required: true, message: '请选择模型类型', trigger: 'change' }
  ],
  hidden_dim: [
    { required: true, message: '请输入隐藏层维度', trigger: 'blur' }
  ],
  num_layers: [
    { required: true, message: '请输入层数', trigger: 'blur' }
  ]
}

// 计算属性
const trainingModels = computed(() => models.value.filter(model => model.status === 'training').length)
const readyModels = computed(() => models.value.filter(model => model.status === 'ready').length)
const failedModels = computed(() => models.value.filter(model => model.status === 'failed').length)

// 方法
const getStatusType = (status) => {
  const types = {
    pending: 'info',
    training: 'warning',
    ready: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

const getStatusText = (status) => {
  const texts = {
    pending: '待训练',
    training: '训练中',
    ready: '就绪',
    failed: '失败'
  }
  return texts[status] || '未知'
}

const getModelColor = (status) => {
  const colors = {
    pending: '#409eff',
    training: '#e6a23c',
    ready: '#67c23a',
    failed: '#f56c6c'
  }
  return colors[status] || '#909399'
}

const formatTime = (time) => {
  return utils.formatTime(time)
}

const refreshModels = async () => {
  loading.value = true
  try {
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      models.value = await api.getUserModels(currentUser)
    }
  } catch (error) {
    console.error('Failed to fetch models:', error)
    ElMessage.error('获取模型列表失败')
  } finally {
    loading.value = false
  }
}

const createModel = async () => {
  if (!createFormRef.value) return
  
  try {
    await createFormRef.value.validate()
    creating.value = true
    
    const currentUser = localStorage.getItem('currentUser')
    if (!currentUser) {
      ElMessage.error('请先登录')
      return
    }
    
    const modelData = {
      ...createForm,
      user: currentUser,
      status: 'pending'
    }
    
    await api.createModel(modelData)
    ElMessage.success('模型创建成功')
    showCreateDialog.value = false
    resetCreateForm()
    refreshModels()
  } catch (error) {
    console.error('Create model failed:', error)
    ElMessage.error('模型创建失败')
  } finally {
    creating.value = false
  }
}

const viewModel = (model) => {
  console.log('View model:', model)
  // 跳转到模型详情页面
}

const editModel = (model) => {
  console.log('Edit model:', model)
  // 跳转到模型编辑页面
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
    refreshModels()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete model:', error)
      ElMessage.error('删除模型失败')
    }
  }
}

const resetCreateForm = () => {
  Object.assign(createForm, {
    name: '',
    description: '',
    model_type: 'lstm',
    hidden_dim: 64,
    num_layers: 2
  })
}

const handleDialogClose = () => {
  showCreateDialog.value = false
  resetCreateForm()
}

// 生命周期
onMounted(() => {
  refreshModels()
})
</script>

<style scoped>
.model-manager-modern {
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

/* 模型统计 */
.models-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  background: white;
  border-radius: 16px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.stat-card:hover {
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

.stat-icon.training {
  background: linear-gradient(135deg, #e6a23c, #f39c12);
}

.stat-icon.ready {
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

/* 模型容器 */
.models-container {
  background: white;
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

.model-card {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 20px;
  transition: all 0.3s ease;
  background: #fafafa;
}

.model-card:hover {
  border-color: #3b82f6;
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}

.model-card.training {
  border-color: #e6a23c;
  background: linear-gradient(135deg, rgba(230, 162, 60, 0.05), rgba(243, 156, 18, 0.05));
}

.model-card.ready {
  border-color: #67c23a;
  background: linear-gradient(135deg, rgba(103, 194, 58, 0.05), rgba(39, 174, 96, 0.05));
}

.model-card.failed {
  border-color: #f56c6c;
  background: linear-gradient(135deg, rgba(245, 108, 108, 0.05), rgba(231, 76, 60, 0.05));
}

.model-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.model-body {
  margin-bottom: 16px;
}

.model-name {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.model-description {
  font-size: 14px;
  color: #6b7280;
  margin: 0 0 12px 0;
  line-height: 1.5;
}

.model-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
}

.info-label {
  color: #6b7280;
  font-weight: 500;
}

.info-value {
  color: #1f2937;
  font-weight: 600;
}

.model-footer {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.progress-label {
  font-size: 12px;
  color: #6b7280;
  font-weight: 500;
}

.progress-percent {
  font-size: 12px;
  color: #1f2937;
  font-weight: 600;
}

.progress-bar {
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

.model-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 80px 20px;
  background: white;
  border-radius: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
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

/* 对话框 */
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .models-grid {
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    gap: 20px;
    align-items: stretch;
  }
  
  .models-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .models-grid {
    grid-template-columns: 1fr;
  }
}
</style>
