<template>
  <div class="task-editor-modern">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <div class="header-left">
          <el-button
            type="text"
            @click="$router.go(-1)"
            class="back-button"
          >
            <el-icon><ArrowLeft /></el-icon>
            返回
          </el-button>
          <div class="header-text">
            <h1 class="page-title">{{ isEdit ? '编辑任务' : '创建新任务' }}</h1>
            <p class="page-subtitle">{{ isEdit ? '修改任务配置和参数' : '配置时间序列预测任务' }}</p>
          </div>
        </div>
        <div class="header-actions">
          <el-button size="large" @click="resetForm">
            <el-icon><Refresh /></el-icon>
            重置
          </el-button>
          <el-button
            type="primary"
            size="large"
            @click="saveTask"
            :loading="saving"
          >
            <el-icon><Check /></el-icon>
            {{ isEdit ? '保存更改' : '创建任务' }}
          </el-button>
        </div>
      </div>
    </div>

    <!-- 表单内容 -->
    <div class="form-container">
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-width="120px"
        class="task-form"
      >
        <!-- 基本信息 -->
        <div class="form-section">
          <div class="section-header">
            <h3 class="section-title">
              <el-icon><InfoFilled /></el-icon>
              基本信息
            </h3>
            <p class="section-description">配置任务的基本信息和描述</p>
          </div>
          
          <div class="form-grid">
            <el-form-item label="任务名称" prop="name" class="form-item-full">
              <el-input
                v-model="form.name"
                placeholder="请输入任务名称"
                size="large"
                clearable
              />
            </el-form-item>
            
            <el-form-item label="任务描述" prop="description" class="form-item-full">
              <el-input
                v-model="form.description"
                type="textarea"
                :rows="3"
                placeholder="请输入任务描述"
                maxlength="500"
                show-word-limit
              />
            </el-form-item>
            
            <el-form-item label="用户" prop="user">
              <el-input
                v-model="form.user"
                placeholder="请输入用户名"
                size="large"
                clearable
              />
            </el-form-item>
            
            <el-form-item label="优先级" prop="priority">
              <el-select v-model="form.priority" placeholder="选择优先级" size="large">
                <el-option label="低" value="low" />
                <el-option label="中" value="medium" />
                <el-option label="高" value="high" />
                <el-option label="紧急" value="urgent" />
              </el-select>
            </el-form-item>
          </div>
        </div>

        <!-- 数据配置 -->
        <div class="form-section">
          <div class="section-header">
            <h3 class="section-title">
              <el-icon><DataAnalysis /></el-icon>
              数据配置
            </h3>
            <p class="section-description">配置数据源和查询参数</p>
          </div>
          
          <div class="form-grid">
            <el-form-item label="Prometheus URL" prop="prometheus_url" class="form-item-full">
              <el-input
                v-model="form.prometheus_url"
                placeholder="http://localhost:9090"
                size="large"
                clearable
              >
                <template #prepend>
                  <el-icon><Link /></el-icon>
                </template>
              </el-input>
            </el-form-item>
            
            <el-form-item label="指标查询" prop="metric_query" class="form-item-full">
              <el-input
                v-model="form.metric_query"
                placeholder="storage_used_bytes{user='alice'}"
                size="large"
                clearable
              >
                <template #prepend>
                  <el-icon><Search /></el-icon>
                </template>
              </el-input>
            </el-form-item>
            
            <el-form-item label="时间范围" prop="time_range">
              <el-select v-model="form.time_range" placeholder="选择时间范围" size="large">
                <el-option label="最近1天" value="1d" />
                <el-option label="最近3天" value="3d" />
                <el-option label="最近7天" value="7d" />
                <el-option label="最近30天" value="30d" />
                <el-option label="自定义" value="custom" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="采样间隔" prop="step">
              <el-select v-model="form.step" placeholder="选择采样间隔" size="large">
                <el-option label="1分钟" value="1m" />
                <el-option label="5分钟" value="5m" />
                <el-option label="15分钟" value="15m" />
                <el-option label="1小时" value="1h" />
                <el-option label="1天" value="1d" />
              </el-select>
            </el-form-item>
          </div>
        </div>

        <!-- 模型配置 -->
        <div class="form-section">
          <div class="section-header">
            <h3 class="section-title">
              <el-icon><Cpu /></el-icon>
              模型配置
            </h3>
            <p class="section-description">配置LSTM模型参数</p>
          </div>
          
          <div class="form-grid">
            <el-form-item label="模型类型" prop="model_type">
              <el-select v-model="form.model_type" placeholder="选择模型类型" size="large">
                <el-option label="LSTM" value="lstm" />
                <el-option label="GRU" value="gru" />
                <el-option label="Transformer" value="transformer" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="序列长度" prop="sequence_length">
              <el-input-number
                v-model="form.sequence_length"
                :min="1"
                :max="100"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="隐藏层维度" prop="hidden_dim">
              <el-input-number
                v-model="form.hidden_dim"
                :min="32"
                :max="512"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="层数" prop="num_layers">
              <el-input-number
                v-model="form.num_layers"
                :min="1"
                :max="6"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="预测步数" prop="prediction_steps">
              <el-input-number
                v-model="form.prediction_steps"
                :min="1"
                :max="24"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="学习率" prop="learning_rate">
              <el-input-number
                v-model="form.learning_rate"
                :min="0.0001"
                :max="0.1"
                :step="0.0001"
                :precision="4"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="批次大小" prop="batch_size">
              <el-input-number
                v-model="form.batch_size"
                :min="1"
                :max="128"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="训练轮数" prop="epochs">
              <el-input-number
                v-model="form.epochs"
                :min="1"
                :max="1000"
                size="large"
                controls-position="right"
              />
            </el-form-item>
          </div>
        </div>

        <!-- 高级配置 -->
        <div class="form-section">
          <div class="section-header">
            <h3 class="section-title">
              <el-icon><Setting /></el-icon>
              高级配置
            </h3>
            <p class="section-description">配置高级参数和选项</p>
          </div>
          
          <div class="form-grid">
            <el-form-item label="使用GPU" prop="use_gpu">
              <el-switch
                v-model="form.use_gpu"
                active-text="启用"
                inactive-text="禁用"
                size="large"
              />
            </el-form-item>
            
            <el-form-item label="早停机制" prop="early_stopping">
              <el-switch
                v-model="form.early_stopping"
                active-text="启用"
                inactive-text="禁用"
                size="large"
              />
            </el-form-item>
            
            <el-form-item label="数据验证比例" prop="validation_split">
              <el-input-number
                v-model="form.validation_split"
                :min="0.1"
                :max="0.5"
                :step="0.05"
                :precision="2"
                size="large"
                controls-position="right"
              />
            </el-form-item>
            
            <el-form-item label="随机种子" prop="random_seed">
              <el-input-number
                v-model="form.random_seed"
                :min="1"
                :max="999999"
                size="large"
                controls-position="right"
              />
            </el-form-item>
          </div>
        </div>
      </el-form>
    </div>

    <!-- 预览面板 -->
    <div class="preview-panel">
      <div class="panel-header">
        <h3 class="panel-title">
          <el-icon><View /></el-icon>
          配置预览
        </h3>
      </div>
      <div class="panel-content">
        <div class="preview-item">
          <span class="preview-label">任务名称:</span>
          <span class="preview-value">{{ form.name || '未设置' }}</span>
        </div>
        <div class="preview-item">
          <span class="preview-label">模型类型:</span>
          <span class="preview-value">{{ form.model_type || '未设置' }}</span>
        </div>
        <div class="preview-item">
          <span class="preview-label">预测步数:</span>
          <span class="preview-value">{{ form.prediction_steps || '未设置' }}</span>
        </div>
        <div class="preview-item">
          <span class="preview-label">使用GPU:</span>
          <span class="preview-value">{{ form.use_gpu ? '是' : '否' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const route = useRoute()
const router = useRouter()

// 响应式数据
const formRef = ref()
const saving = ref(false)
const isEdit = computed(() => !!route.params.id)

const form = reactive({
  name: '',
  description: '',
  user: '',
  priority: 'medium',
  prometheus_url: 'http://localhost:9090',
  metric_query: '',
  time_range: '7d',
  step: '1h',
  model_type: 'lstm',
  sequence_length: 24,
  hidden_dim: 64,
  num_layers: 2,
  prediction_steps: 1,
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 100,
  use_gpu: false,
  early_stopping: true,
  validation_split: 0.2,
  random_seed: 42
})

const rules = {
  name: [
    { required: true, message: '请输入任务名称', trigger: 'blur' },
    { min: 2, max: 50, message: '任务名称长度在 2 到 50 个字符', trigger: 'blur' }
  ],
  user: [
    { required: true, message: '请输入用户名', trigger: 'blur' }
  ],
  prometheus_url: [
    { required: true, message: '请输入Prometheus URL', trigger: 'blur' },
    { type: 'url', message: '请输入正确的URL格式', trigger: 'blur' }
  ],
  metric_query: [
    { required: true, message: '请输入指标查询', trigger: 'blur' }
  ],
  model_type: [
    { required: true, message: '请选择模型类型', trigger: 'change' }
  ],
  sequence_length: [
    { required: true, message: '请输入序列长度', trigger: 'blur' }
  ],
  prediction_steps: [
    { required: true, message: '请输入预测步数', trigger: 'blur' }
  ]
}

// 方法
const resetForm = () => {
  if (formRef.value) {
    formRef.value.resetFields()
  }
}

const saveTask = async () => {
  if (!formRef.value) return
  
  try {
    await formRef.value.validate()
    saving.value = true
    
    const currentUser = localStorage.getItem('currentUser')
    if (!currentUser) {
      ElMessage.error('请先登录')
      return
    }
    
    const taskData = {
      ...form,
      user: currentUser,
      status: 'pending'
    }
    
    if (isEdit.value) {
      await api.updateTask(route.params.id, taskData)
      ElMessage.success('任务更新成功')
    } else {
      await api.createTask(taskData)
      ElMessage.success('任务创建成功')
    }
    
    router.push('/tasks')
  } catch (error) {
    console.error('Save task failed:', error)
    ElMessage.error(isEdit.value ? '任务更新失败' : '任务创建失败')
  } finally {
    saving.value = false
  }
}

const loadTask = async () => {
  if (isEdit.value) {
    try {
      const task = await api.getTask(route.params.id)
      Object.assign(form, task)
    } catch (error) {
      console.error('Load task failed:', error)
      ElMessage.error('加载任务失败')
    }
  } else {
    // 设置默认用户
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      form.user = currentUser
    }
  }
}

// 生命周期
onMounted(() => {
  loadTask()
})
</script>

<style scoped>
.task-editor-modern {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  display: flex;
  flex-direction: column;
}

/* 页面头部 */
.page-header {
  background: white;
  border-radius: 20px;
  padding: 24px 32px;
  margin-bottom: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 20px;
}

.back-button {
  font-size: 16px;
  color: #6b7280;
  padding: 8px 16px;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.back-button:hover {
  background: #f3f4f6;
  color: #374151;
}

.header-text {
  flex: 1;
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 4px 0;
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

/* 表单容器 */
.form-container {
  flex: 1;
  display: flex;
  gap: 24px;
}

.task-form {
  flex: 1;
  background: white;
  border-radius: 20px;
  padding: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  max-height: calc(100vh - 200px);
  overflow-y: auto;
}

/* 表单区块 */
.form-section {
  margin-bottom: 40px;
}

.section-header {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 2px solid #f3f4f6;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 20px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.section-title .el-icon {
  color: #3b82f6;
}

.section-description {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

/* 表单网格 */
.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.form-item-full {
  grid-column: 1 / -1;
}

/* 预览面板 */
.preview-panel {
  width: 300px;
  background: white;
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  height: fit-content;
  position: sticky;
  top: 24px;
}

.panel-header {
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e5e7eb;
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.panel-title .el-icon {
  color: #3b82f6;
}

.panel-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.preview-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
}

.preview-label {
  font-size: 14px;
  color: #6b7280;
  font-weight: 500;
}

.preview-value {
  font-size: 14px;
  color: #1f2937;
  font-weight: 600;
  text-align: right;
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .form-container {
    flex-direction: column;
  }
  
  .preview-panel {
    width: 100%;
    position: static;
  }
  
  .form-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }
  
  .header-left {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }
  
  .header-actions {
    justify-content: stretch;
  }
  
  .header-actions .el-button {
    flex: 1;
  }
  
  .task-form {
    padding: 20px;
  }
  
  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
