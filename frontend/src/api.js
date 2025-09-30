/**
 * API 通信模块
 * 封装与后端 FastAPI 服务的通信
 */

import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建 axios 实例
const request = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
request.interceptors.request.use(
  config => {
    // 可以在这里添加认证 token
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
request.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    console.error('Response error:', error)
    
    if (error.response) {
      const { status, data } = error.response
      
      switch (status) {
        case 400:
          ElMessage.error(data.detail || '请求参数错误')
          break
        case 401:
          ElMessage.error('未授权，请重新登录')
          break
        case 403:
          ElMessage.error('权限不足')
          break
        case 404:
          ElMessage.error('资源不存在')
          break
        case 500:
          ElMessage.error('服务器内部错误')
          break
        default:
          ElMessage.error(data.detail || '请求失败')
      }
    } else if (error.request) {
      ElMessage.error('网络连接失败，请检查网络')
    } else {
      ElMessage.error('请求配置错误')
    }
    
    return Promise.reject(error)
  }
)

// API 接口定义
export const api = {
  // 用户管理
  async createUser(username, email) {
    return request.post('/users', { username, email })
  },

  async getUser(userId) {
    return request.get(`/users/${userId}`)
  },

  async getUserByUsername(username) {
    return request.get(`/users/username/${username}`)
  },

  // 任务管理
  async createTask(taskData) {
    return request.post('/tasks', taskData)
  },

  async getTask(taskId) {
    return request.get(`/tasks/${taskId}`)
  },

  async getUserTasks(username) {
    return request.get(`/tasks/user/${username}`)
  },

  async deleteTask(taskId) {
    return request.delete(`/tasks/${taskId}`)
  },

  // 模型管理
  async getModel(modelId) {
    return request.get(`/models/${modelId}`)
  },

  async getUserModels(username) {
    return request.get(`/models/user/${username}`)
  },

  async getTaskModels(taskId) {
    return request.get(`/models/task/${taskId}`)
  },

  // 预测服务
  async predict(predictionData) {
    return request.post('/predict', predictionData)
  },

  async getMetrics(user, hours = 24) {
    return request.get(`/data/metrics/${user}?hours=${hours}`)
  },

  // 系统信息
  async getStats() {
    return request.get('/stats')
  },

  async healthCheck() {
    return request.get('/health')
  }
}

// 工具函数
export const utils = {
  // 格式化时间
  formatTime(timestamp) {
    return new Date(timestamp).toLocaleString('zh-CN')
  },

  // 格式化文件大小
  formatFileSize(bytes) {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  },

  // 格式化百分比
  formatPercentage(value, total) {
    if (total === 0) return '0%'
    return ((value / total) * 100).toFixed(1) + '%'
  },

  // 获取状态颜色
  getStatusColor(status) {
    const colors = {
      pending: '#909399',
      running: '#409eff',
      completed: '#67c23a',
      failed: '#f56c6c',
      training: '#e6a23c'
    }
    return colors[status] || '#909399'
  },

  // 获取状态文本
  getStatusText(status) {
    const texts = {
      pending: '等待中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      training: '训练中'
    }
    return texts[status] || status
  },

  // 生成随机颜色
  generateColor() {
    const colors = [
      '#409eff', '#67c23a', '#e6a23c', '#f56c6c', '#909399',
      '#c71585', '#ff6347', '#32cd32', '#ffd700', '#ff69b4'
    ]
    return colors[Math.floor(Math.random() * colors.length)]
  },

  // 防抖函数
  debounce(func, wait) {
    let timeout
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout)
        func(...args)
      }
      clearTimeout(timeout)
      timeout = setTimeout(later, wait)
    }
  },

  // 节流函数
  throttle(func, limit) {
    let inThrottle
    return function() {
      const args = arguments
      const context = this
      if (!inThrottle) {
        func.apply(context, args)
        inThrottle = true
        setTimeout(() => inThrottle = false, limit)
      }
    }
  }
}

export default request
