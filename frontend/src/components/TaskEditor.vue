<template>
  <div class="task-editor">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>{{ isEdit ? '编辑任务' : '创建任务' }}</h1>
      <p>{{ isEdit ? '修改时间序列预测任务' : '创建新的时间序列预测任务' }}</p>
    </div>

    <!-- 任务表单 -->
    <el-card class="form-card">
      <el-form
        ref="taskFormRef"
        :model="taskForm"
        :rules="taskRules"
        label-width="120px"
        size="large"
      >
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="任务名称" prop="name">
              <el-input
                v-model="taskForm.name"
                placeholder="请输入任务名称"
                clearable
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="用户" prop="user">
              <el-input
                v-model="taskForm.user"
                placeholder="请输入用户名"
                clearable
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="指标查询" prop="metric_query">
          <el-input
            v-model="taskForm.metric_query"
            type="textarea"
            :rows="3"
            placeholder="请输入 PromQL 查询语句，例如: storage_used_bytes{user='alice'}"
            clearable
          />
          <div class="form-tip">
            <el-icon><InfoFilled /></el-icon>
            支持 PromQL 查询语法，可以使用标签过滤
          </div>
        </el-form-item>

        <!-- 模型配置 -->
        <div class="config-section">
          <h3>模型配置</h3>
          
          <el-row :gutter="20">
            <el-col :span="8">
              <el-form-item label="序列长度" prop="config.sequence_length">
                <el-input-number
                  v-model="taskForm.config.sequence_length"
                  :min="1"
                  :max="168"
                  placeholder="序列长度"
                  style="width: 100%"
                />
                <div class="form-tip">使用过去N小时的数据预测未来</div>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="预测步数" prop="config.prediction_steps">
                <el-input-number
                  v-model="taskForm.config.prediction_steps"
                  :min="1"
                  :max="24"
                  placeholder="预测步数"
                  style="width: 100%"
                />
                <div class="form-tip">预测未来N小时的值</div>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="训练轮数" prop="config.epochs">
                <el-input-number
                  v-model="taskForm.config.epochs"
                  :min="10"
                  :max="1000"
                  placeholder="训练轮数"
                  style="width: 100%"
                />
                <div class="form-tip">模型训练迭代次数</div>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :span="8">
              <el-form-item label="隐藏层维度" prop="config.hidden_dim">
                <el-input-number
                  v-model="taskForm.config.hidden_dim"
                  :min="16"
                  :max="512"
                  placeholder="隐藏层维度"
                  style="width: 100%"
                />
                <div class="form-tip">LSTM 隐藏层大小</div>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="LSTM 层数" prop="config.num_layers">
                <el-input-number
                  v-model="taskForm.config.num_layers"
                  :min="1"
                  :max="5"
                  placeholder="LSTM 层数"
                  style="width: 100%"
                />
                <div class="form-tip">LSTM 网络层数</div>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="学习率" prop="config.learning_rate">
                <el-input-number
                  v-model="taskForm.config.learning_rate"
                  :min="0.0001"
                  :max="0.1"
                  :step="0.0001"
                  :precision="4"
                  placeholder="学习率"
                  style="width: 100%"
                />
                <div class="form-tip">模型训练学习率</div>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :span="8">
              <el-form-item label="批次大小" prop="config.batch_size">
                <el-input-number
                  v-model="taskForm.config.batch_size"
                  :min="1"
                  :max="128"
                  placeholder="批次大小"
                  style="width: 100%"
                />
                <div class="form-tip">训练批次大小</div>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="Dropout" prop="config.dropout">
                <el-input-number
                  v-model="taskForm.config.dropout"
                  :min="0"
                  :max="0.9"
                  :step="0.1"
                  :precision="1"
                  placeholder="Dropout"
                  style="width: 100%"
                />
                <div class="form-tip">防止过拟合的 Dropout 比例</div>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="早停耐心值" prop="config.early_stopping_patience">
                <el-input-number
                  v-model="taskForm.config.early_stopping_patience"
                  :min="5"
                  :max="50"
                  placeholder="早停耐心值"
                  style="width: 100%"
                />
                <div class="form-tip">早停机制的耐心值</div>
              </el-form-item>
            </el-col>
          </el-row>
        </div>

        <!-- 高级配置 -->
        <div class="config-section">
          <h3>高级配置</h3>
          
          <el-form-item label="数据预处理">
            <el-checkbox-group v-model="taskForm.config.preprocessing">
              <el-checkbox label="normalize">数据归一化</el-checkbox>
              <el-checkbox label="fill_missing">填充缺失值</el-checkbox>
              <el-checkbox label="remove_outliers">移除异常值</el-checkbox>
            </el-checkbox-group>
          </el-form-item>

          <el-form-item label="验证集比例">
            <el-slider
              v-model="taskForm.config.validation_split"
              :min="0.1"
              :max="0.5"
              :step="0.05"
              :format-tooltip="formatValidationSplit"
              style="width: 100%"
            />
          </el-form-item>
        </div>

        <!-- 操作按钮 -->
        <el-form-item>
          <div class="form-actions">
            <el-button @click="handleCancel">取消</el-button>
            <el-button @click="handleReset">重置</el-button>
            <el-button
              type="primary"
              @click="handleSubmit"
              :loading="submitting"
            >
              {{ isEdit ? '更新任务' : '创建任务' }}
            </el-button>
          </div>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 配置预览 -->
    <el-card class="preview-card" v-if="showPreview">
      <template #header>
        <div class="card-header">
          <span>配置预览</span>
          <el-button type="text" @click="showPreview = false">
            <el-icon><Close /></el-icon>
          </el-button>
        </div>
      </template>
      <pre class="config-preview">{{ JSON.stringify(taskForm, null, 2) }}</pre>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '../api'

// 路由
const route = useRoute()
const router = useRouter()

// 响应式数据
const taskFormRef = ref()
const submitting = ref(false)
const showPreview = ref(false)

// 计算属性
const isEdit = computed(() => !!route.params.id)

// 表单数据
const taskForm = reactive({
  name: '',
  user: '',
  metric_query: '',
  config: {
    sequence_length: 24,
    prediction_steps: 1,
    epochs: 100,
    hidden_dim: 64,
    num_layers: 2,
    learning_rate: 0.001,
    batch_size: 32,
    dropout: 0.2,
    early_stopping_patience: 10,
    preprocessing: ['normalize', 'fill_missing'],
    validation_split: 0.2
  }
})

// 表单验证规则
const taskRules = {
  name: [
    { required: true, message: '请输入任务名称', trigger: 'blur' },
    { min: 2, max: 50, message: '任务名称长度在 2 到 50 个字符', trigger: 'blur' }
  ],
  user: [
    { required: true, message: '请输入用户名', trigger: 'blur' }
  ],
  metric_query: [
    { required: true, message: '请输入指标查询语句', trigger: 'blur' },
    { min: 10, message: '指标查询语句至少 10 个字符', trigger: 'blur' }
  ],
  'config.sequence_length': [
    { required: true, message: '请输入序列长度', trigger: 'blur' }
  ],
  'config.prediction_steps': [
    { required: true, message: '请输入预测步数', trigger: 'blur' }
  ],
  'config.epochs': [
    { required: true, message: '请输入训练轮数', trigger: 'blur' }
  ]
}

// 方法
const formatValidationSplit = (value) => {
  return `${(value * 100).toFixed(0)}%`
}

const loadTask = async () => {
  if (!isEdit.value) return

  try {
    const task = await api.getTask(route.params.id)
    Object.assign(taskForm, {
      name: task.name,
      user: task.user,
      metric_query: task.metric_query,
      config: { ...taskForm.config, ...task.config }
    })
  } catch (error) {
    console.error('Failed to load task:', error)
    ElMessage.error('加载任务失败')
    router.push('/tasks')
  }
}

const handleSubmit = async () => {
  if (!taskFormRef.value) return

  try {
    await taskFormRef.value.validate()
    submitting.value = true

    if (isEdit.value) {
      // 更新任务
      await api.updateTask(route.params.id, taskForm)
      ElMessage.success('任务更新成功')
    } else {
      // 创建任务
      await api.createTask(taskForm)
      ElMessage.success('任务创建成功')
    }

    router.push('/tasks')

  } catch (error) {
    console.error('Submit failed:', error)
    ElMessage.error(isEdit.value ? '更新任务失败' : '创建任务失败')
  } finally {
    submitting.value = false
  }
}

const handleCancel = () => {
  router.push('/tasks')
}

const handleReset = () => {
  if (taskFormRef.value) {
    taskFormRef.value.resetFields()
  }
  
  // 重置为默认值
  Object.assign(taskForm, {
    name: '',
    user: '',
    metric_query: '',
    config: {
      sequence_length: 24,
      prediction_steps: 1,
      epochs: 100,
      hidden_dim: 64,
      num_layers: 2,
      learning_rate: 0.001,
      batch_size: 32,
      dropout: 0.2,
      early_stopping_patience: 10,
      preprocessing: ['normalize', 'fill_missing'],
      validation_split: 0.2
    }
  })
}

// 生命周期
onMounted(() => {
  // 设置默认用户
  const currentUser = localStorage.getItem('currentUser')
  if (currentUser) {
    taskForm.user = currentUser
  }

  // 如果是编辑模式，加载任务数据
  if (isEdit.value) {
    loadTask()
  }
})
</script>

<style scoped>
.task-editor {
  max-width: 1000px;
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

.form-card {
  margin-bottom: 20px;
}

.config-section {
  margin: 32px 0;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.config-section h3 {
  font-size: 18px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 20px;
  padding-bottom: 8px;
  border-bottom: 2px solid #409eff;
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.form-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 20px;
}

.preview-card {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.config-preview {
  background-color: #f5f5f5;
  padding: 16px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.4;
  color: #606266;
  max-height: 400px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .task-editor {
    padding: 0 16px;
  }
  
  .config-section {
    padding: 16px;
  }
  
  .form-actions {
    flex-direction: column;
  }
}
</style>
