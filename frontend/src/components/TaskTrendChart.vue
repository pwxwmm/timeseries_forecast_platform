<template>
  <div class="task-trend-chart">
    <div class="chart-header">
      <div class="header-left">
        <h3 class="chart-title">
          <el-icon><TrendCharts /></el-icon>
          任务趋势分析
        </h3>
        <p class="chart-subtitle">任务创建和完成趋势</p>
      </div>
      <div class="header-right">
        <el-radio-group v-model="timeRange" @change="handleTimeRangeChange" size="small">
          <el-radio-button label="7d">7天</el-radio-button>
          <el-radio-button label="30d">30天</el-radio-button>
          <el-radio-button label="90d">90天</el-radio-button>
        </el-radio-group>
      </div>
    </div>
    
    <div class="chart-container">
      <div class="chart" v-loading="loading">
        <svg class="trend-svg" viewBox="0 0 800 300" preserveAspectRatio="xMidYMid meet">
          <!-- 渐变定义 -->
          <defs>
            <linearGradient id="createdGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#667eea;stop-opacity:0.8" />
              <stop offset="100%" style="stop-color:#667eea;stop-opacity:0.1" />
            </linearGradient>
            <linearGradient id="completedGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#10b981;stop-opacity:0.8" />
              <stop offset="100%" style="stop-color:#10b981;stop-opacity:0.1" />
            </linearGradient>
            <linearGradient id="pendingGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0.8" />
              <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:0.1" />
            </linearGradient>
          </defs>
          
          <!-- 网格线 -->
          <g class="grid-lines">
            <line v-for="i in 5" :key="i" 
                  :x1="80" :y1="60 + (i-1) * 48" 
                  :x2="720" :y2="60 + (i-1) * 48" 
                  stroke="#f3f4f6" stroke-width="1" stroke-dasharray="2,2" />
          </g>
          
          <!-- 创建任务曲线 -->
          <path :d="createdPath" 
                fill="url(#createdGradient)" 
                class="trend-line created-line" />
          <path :d="createdLinePath" 
                stroke="#667eea" 
                stroke-width="3" 
                fill="none" 
                class="trend-line created-line" />
          
          <!-- 完成任务曲线 -->
          <path :d="completedPath" 
                fill="url(#completedGradient)" 
                class="trend-line completed-line" />
          <path :d="completedLinePath" 
                stroke="#10b981" 
                stroke-width="3" 
                fill="none" 
                class="trend-line completed-line" />
          
          <!-- 待处理任务曲线 -->
          <path :d="pendingPath" 
                fill="url(#pendingGradient)" 
                class="trend-line pending-line" />
          <path :d="pendingLinePath" 
                stroke="#f59e0b" 
                stroke-width="3" 
                fill="none" 
                class="trend-line pending-line" />
          
          <!-- 数据点 -->
          <g class="data-points">
            <circle v-for="(point, index) in createdPoints" 
                    :key="`created-${index}`"
                    :cx="point.x" :cy="point.y" 
                    r="4" 
                    fill="#667eea" 
                    stroke="#fff" 
                    stroke-width="2"
                    class="data-point created-point" />
            <circle v-for="(point, index) in completedPoints" 
                    :key="`completed-${index}`"
                    :cx="point.x" :cy="point.y" 
                    r="4" 
                    fill="#10b981" 
                    stroke="#fff" 
                    stroke-width="2"
                    class="data-point completed-point" />
            <circle v-for="(point, index) in pendingPoints" 
                    :key="`pending-${index}`"
                    :cx="point.x" :cy="point.y" 
                    r="4" 
                    fill="#f59e0b" 
                    stroke="#fff" 
                    stroke-width="2"
                    class="data-point pending-point" />
          </g>
          
          <!-- X轴标签 -->
          <g class="x-axis-labels">
            <text v-for="(label, index) in xAxisLabels" 
                  :key="index"
                  :x="80 + index * 80" 
                  y="280" 
                  text-anchor="middle" 
                  class="axis-label">{{ label }}</text>
          </g>
          
          <!-- Y轴标签 -->
          <g class="y-axis-labels">
            <text v-for="(label, index) in yAxisLabels" 
                  :key="index"
                  x="70" 
                  :y="60 + index * 48 + 5" 
                  text-anchor="end" 
                  class="axis-label">{{ label }}</text>
          </g>
        </svg>
      </div>
    </div>
    
    <div class="chart-stats">
      <div class="stat-item">
        <div class="stat-icon created">
          <el-icon><Plus /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ totalCreated }}</div>
          <div class="stat-label">总创建</div>
          <div class="stat-trend" :class="createdTrendClass">
            <el-icon><TrendCharts /></el-icon>
            <span>{{ createdTrend }}%</span>
          </div>
        </div>
      </div>
      
      <div class="stat-item">
        <div class="stat-icon completed">
          <el-icon><Check /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ totalCompleted }}</div>
          <div class="stat-label">总完成</div>
          <div class="stat-trend" :class="completedTrendClass">
            <el-icon><TrendCharts /></el-icon>
            <span>{{ completedTrend }}%</span>
          </div>
        </div>
      </div>
      
      <div class="stat-item">
        <div class="stat-icon rate">
          <el-icon><PieChart /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ completionRate }}%</div>
          <div class="stat-label">完成率</div>
          <div class="stat-trend" :class="rateTrendClass">
            <el-icon><TrendCharts /></el-icon>
            <span>{{ rateTrend }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

// 响应式数据
const loading = ref(false)
const timeRange = ref('30d')

// 统计数据
const totalCreated = ref(0)
const totalCompleted = ref(0)
const completionRate = ref(0)
const createdTrend = ref(0)
const completedTrend = ref(0)
const rateTrend = ref(0)

// 图表数据
const chartData = ref([])
const createdPoints = ref([])
const completedPoints = ref([])
const pendingPoints = ref([])
const xAxisLabels = ref([])
const yAxisLabels = ref(['0', '5', '10', '15', '20'])

// 计算属性
const createdTrendClass = computed(() => createdTrend.value >= 0 ? 'positive' : 'negative')
const completedTrendClass = computed(() => completedTrend.value >= 0 ? 'positive' : 'negative')
const rateTrendClass = computed(() => rateTrend.value >= 0 ? 'positive' : 'negative')

// SVG 路径计算
const createdPath = computed(() => {
  if (createdPoints.value.length === 0) return ''
  const points = createdPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cp1x = prev.x + (curr.x - prev.x) / 3
    const cp1y = prev.y
    const cp2x = curr.x - (curr.x - prev.x) / 3
    const cp2y = curr.y
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
  }
  path += ` L ${points[points.length - 1].x} 252 L 80 252 Z`
  return path
})

const createdLinePath = computed(() => {
  if (createdPoints.value.length === 0) return ''
  const points = createdPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cp1x = prev.x + (curr.x - prev.x) / 3
    const cp1y = prev.y
    const cp2x = curr.x - (curr.x - prev.x) / 3
    const cp2y = curr.y
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
  }
  return path
})

const completedPath = computed(() => {
  if (completedPoints.value.length === 0) return ''
  const points = completedPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cp1x = prev.x + (curr.x - prev.x) / 3
    const cp1y = prev.y
    const cp2x = curr.x - (curr.x - prev.x) / 3
    const cp2y = curr.y
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
  }
  path += ` L ${points[points.length - 1].x} 252 L 80 252 Z`
  return path
})

const completedLinePath = computed(() => {
  if (completedPoints.value.length === 0) return ''
  const points = completedPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cp1x = prev.x + (curr.x - prev.x) / 3
    const cp1y = prev.y
    const cp2x = curr.x - (curr.x - prev.x) / 3
    const cp2y = curr.y
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
  }
  return path
})

const pendingPath = computed(() => {
  if (pendingPoints.value.length === 0) return ''
  const points = pendingPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cp1x = prev.x + (curr.x - prev.x) / 3
    const cp1y = prev.y
    const cp2x = curr.x - (curr.x - prev.x) / 3
    const cp2y = curr.y
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
  }
  path += ` L ${points[points.length - 1].x} 252 L 80 252 Z`
  return path
})

const pendingLinePath = computed(() => {
  if (pendingPoints.value.length === 0) return ''
  const points = pendingPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cp1x = prev.x + (curr.x - prev.x) / 3
    const cp1y = prev.y
    const cp2x = curr.x - (curr.x - prev.x) / 3
    const cp2y = curr.y
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
  }
  return path
})

// 模拟数据生成
const generateMockData = (days = 30) => {
  const data = []
  const now = new Date()
  const labels = []
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(date.getDate() - i)
    
    // 模拟任务创建数据（有周期性）
    const dayOfWeek = date.getDay()
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6
    const baseCreated = isWeekend ? 2 : 8
    const created = Math.max(0, baseCreated + Math.floor(Math.random() * 6) - 2)
    
    // 模拟任务完成数据（通常比创建少一些）
    const baseCompleted = Math.floor(created * (0.7 + Math.random() * 0.2))
    const completed = Math.max(0, baseCompleted)
    
    data.push({
      date: date.toISOString().split('T')[0],
      created,
      completed,
      pending: Math.max(0, created - completed)
    })
    
    // 生成 X 轴标签
    if (days <= 7) {
      labels.push(`${date.getMonth() + 1}/${date.getDate()}`)
    } else if (days <= 30) {
      if (i % 5 === 0) {
        labels.push(`${date.getMonth() + 1}/${date.getDate()}`)
      } else {
        labels.push('')
      }
    } else {
      if (i % 10 === 0) {
        labels.push(`${date.getMonth() + 1}/${date.getDate()}`)
      } else {
        labels.push('')
      }
    }
  }
  
  xAxisLabels.value = labels
  return data
}

// 计算统计数据
const calculateStats = (data) => {
  totalCreated.value = data.reduce((sum, item) => sum + item.created, 0)
  totalCompleted.value = data.reduce((sum, item) => sum + item.completed, 0)
  completionRate.value = totalCreated.value > 0 ? Math.round((totalCompleted.value / totalCreated.value) * 100) : 0
  
  // 计算趋势（与前期对比）
  const midPoint = Math.floor(data.length / 2)
  const firstHalf = data.slice(0, midPoint)
  const secondHalf = data.slice(midPoint)
  
  const firstHalfCreated = firstHalf.reduce((sum, item) => sum + item.created, 0)
  const secondHalfCreated = secondHalf.reduce((sum, item) => sum + item.created, 0)
  createdTrend.value = firstHalfCreated > 0 ? Math.round(((secondHalfCreated - firstHalfCreated) / firstHalfCreated) * 100) : 0
  
  const firstHalfCompleted = firstHalf.reduce((sum, item) => sum + item.completed, 0)
  const secondHalfCompleted = secondHalf.reduce((sum, item) => sum + item.completed, 0)
  completedTrend.value = firstHalfCompleted > 0 ? Math.round(((secondHalfCompleted - firstHalfCompleted) / firstHalfCompleted) * 100) : 0
  
  const firstHalfRate = firstHalfCreated > 0 ? (firstHalfCompleted / firstHalfCreated) * 100 : 0
  const secondHalfRate = secondHalfCreated > 0 ? (secondHalfCompleted / secondHalfCreated) * 100 : 0
  rateTrend.value = firstHalfRate > 0 ? Math.round(secondHalfRate - firstHalfRate) : 0
}

// 生成图表点数据
const generateChartPoints = (data) => {
  const maxValue = Math.max(...data.map(item => Math.max(item.created, item.completed, item.pending)))
  const scale = maxValue > 0 ? 192 / maxValue : 1 // 192 是图表高度
  
  const newCreatedPoints = []
  const newCompletedPoints = []
  const newPendingPoints = []
  
  data.forEach((item, index) => {
    const x = 80 + (index * 640) / (data.length - 1)
    
    newCreatedPoints.push({
      x,
      y: 252 - item.created * scale
    })
    
    newCompletedPoints.push({
      x,
      y: 252 - item.completed * scale
    })
    
    newPendingPoints.push({
      x,
      y: 252 - item.pending * scale
    })
  })
  
  createdPoints.value = newCreatedPoints
  completedPoints.value = newCompletedPoints
  pendingPoints.value = newPendingPoints
}

// 初始化图表
const initChart = () => {
  const days = timeRange.value === '7d' ? 7 : timeRange.value === '30d' ? 30 : 90
  const data = generateMockData(days)
  calculateStats(data)
  generateChartPoints(data)
}

// 处理时间范围变化
const handleTimeRangeChange = () => {
  loading.value = true
  setTimeout(() => {
    initChart()
    loading.value = false
  }, 500)
}

// 生命周期
onMounted(() => {
  initChart()
})
</script>

<style scoped>
.task-trend-chart {
  background: white;
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.task-trend-chart:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f3f4f6;
}

.header-left {
  flex: 1;
}

.chart-title {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 4px 0;
}

.chart-title .el-icon {
  color: #3b82f6;
}

.chart-subtitle {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

.header-right {
  flex-shrink: 0;
}

.chart-container {
  margin-bottom: 24px;
}

.chart {
  width: 100%;
  height: 300px;
  position: relative;
  overflow: hidden;
}

.trend-svg {
  width: 100%;
  height: 100%;
}

/* 图表动画 */
.trend-line {
  animation: drawLine 2s ease-in-out;
  animation-fill-mode: both;
}

.created-line {
  animation-delay: 0.2s;
}

.completed-line {
  animation-delay: 0.6s;
}

.pending-line {
  animation-delay: 1s;
}

.data-point {
  animation: fadeInScale 0.8s ease-out;
  animation-fill-mode: both;
}

.created-point {
  animation-delay: 0.4s;
}

.completed-point {
  animation-delay: 0.8s;
}

.pending-point {
  animation-delay: 1.2s;
}

.data-point:hover {
  r: 6;
  transition: r 0.3s ease;
}

@keyframes drawLine {
  0% {
    stroke-dasharray: 1000;
    stroke-dashoffset: 1000;
  }
  100% {
    stroke-dasharray: 1000;
    stroke-dashoffset: 0;
  }
}

@keyframes fadeInScale {
  0% {
    opacity: 0;
    transform: scale(0);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.2);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}

.chart-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
  transition: all 0.3s ease;
}

.stat-item:hover {
  background: #e2e8f0;
  transform: translateY(-2px);
}

.stat-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: white;
  flex-shrink: 0;
}

.stat-icon.created {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.stat-icon.completed {
  background: linear-gradient(135deg, #10b981, #059669);
}

.stat-icon.rate {
  background: linear-gradient(135deg, #f59e0b, #d97706);
}

.stat-content {
  flex: 1;
  min-width: 0;
}

.stat-value {
  font-size: 20px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 4px;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
}

.stat-trend.positive {
  background: rgba(16, 185, 129, 0.1);
  color: #059669;
}

.stat-trend.negative {
  background: rgba(239, 68, 68, 0.1);
  color: #dc2626;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .chart-header {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }
  
  .chart-stats {
    grid-template-columns: 1fr;
  }
  
  .chart {
    height: 250px;
  }
}
</style>
