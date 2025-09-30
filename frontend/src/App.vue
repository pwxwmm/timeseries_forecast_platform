<template>
  <div id="app">
    <el-container class="app-container">
      <!-- 侧边栏 -->
      <el-aside class="sidebar" :width="sidebarWidth">
        <div class="sidebar-header">
          <div class="logo">
            <div class="logo-icon">
              <el-icon><TrendCharts /></el-icon>
            </div>
            <div class="logo-text" v-show="!isCollapsed">
              <div class="logo-title">TimeSeries AI</div>
              <div class="logo-subtitle">预测平台</div>
            </div>
          </div>
          <el-button 
            class="collapse-btn" 
            @click="toggleSidebar"
            :icon="isCollapsed ? 'Expand' : 'Fold'"
            circle
            size="small"
          />
        </div>
        
        <el-menu
          :default-active="$route.path"
          :collapse="isCollapsed"
          router
          class="sidebar-menu"
          background-color="transparent"
          text-color="#8b9dc3"
          active-text-color="#ffffff"
        >
          <el-menu-item index="/" class="menu-item">
            <el-icon><House /></el-icon>
            <template #title>
              <span>仪表板</span>
            </template>
          </el-menu-item>
          <el-menu-item index="/tasks" class="menu-item">
            <el-icon><List /></el-icon>
            <template #title>
              <span>任务管理</span>
            </template>
          </el-menu-item>
          <el-menu-item index="/models" class="menu-item">
            <el-icon><Cpu /></el-icon>
            <template #title>
              <span>模型管理</span>
            </template>
          </el-menu-item>
        </el-menu>
        
        <div class="sidebar-footer">
          <el-dropdown trigger="click" placement="top-start">
            <div class="user-profile">
              <el-avatar :size="isCollapsed ? 32 : 40" class="user-avatar">
                <el-icon><User /></el-icon>
              </el-avatar>
              <div class="user-info" v-show="!isCollapsed">
                <div class="user-name">{{ currentUser || '未登录' }}</div>
                <div class="user-role">管理员</div>
              </div>
            </div>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item @click="showUserDialog = true">
                  <el-icon><User /></el-icon>
                  用户信息
                </el-dropdown-item>
                <el-dropdown-item @click="logout">
                  <el-icon><SwitchButton /></el-icon>
                  退出登录
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-aside>

      <!-- 主内容区域 -->
      <el-container class="main-container">
        <el-header class="main-header">
          <div class="header-left">
            <el-breadcrumb separator="/">
              <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
              <el-breadcrumb-item v-if="$route.name">{{ getPageTitle() }}</el-breadcrumb-item>
            </el-breadcrumb>
          </div>
          <div class="header-right">
            <el-button-group>
              <el-button :icon="'Refresh'" circle @click="refreshPage" />
              <el-button :icon="'Setting'" circle @click="showSettings = true" />
            </el-button-group>
          </div>
        </el-header>
        
        <el-main class="main-content">
          <div class="content-wrapper">
            <router-view />
          </div>
        </el-main>
      </el-container>
    </el-container>

    <!-- 用户登录对话框 -->
    <el-dialog
      v-model="showUserDialog"
      title="用户登录"
      width="400px"
      :before-close="handleUserDialogClose"
    >
      <el-form
        ref="userFormRef"
        :model="userForm"
        :rules="userRules"
        label-width="80px"
      >
        <el-form-item label="用户名" prop="username">
          <el-input
            v-model="userForm.username"
            placeholder="请输入用户名"
            clearable
          />
        </el-form-item>
        <el-form-item label="邮箱" prop="email">
          <el-input
            v-model="userForm.email"
            placeholder="请输入邮箱"
            clearable
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showUserDialog = false">取消</el-button>
          <el-button type="primary" @click="handleLogin">登录</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from './api'

// 响应式数据
const currentUser = ref('')
const showUserDialog = ref(false)
const showSettings = ref(false)
const isCollapsed = ref(false)
const userFormRef = ref()

const userForm = reactive({
  username: '',
  email: ''
})

const userRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' }
  ],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '请输入正确的邮箱格式', trigger: 'blur' }
  ]
}

// 计算属性
const sidebarWidth = computed(() => isCollapsed.value ? '64px' : '260px')

// 页面标题映射
const pageTitles = {
  '/': '仪表板',
  '/tasks': '任务管理',
  '/models': '模型管理'
}

// 方法
const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const getPageTitle = () => {
  return pageTitles[window.location.pathname] || '未知页面'
}

const refreshPage = () => {
  window.location.reload()
}

const handleLogin = async () => {
  if (!userFormRef.value) return
  
  try {
    await userFormRef.value.validate()
    
    // 创建或获取用户
    const response = await api.createUser(userForm.username, userForm.email)
    currentUser.value = userForm.username
    
    // 保存到本地存储
    localStorage.setItem('currentUser', userForm.username)
    localStorage.setItem('userId', response.id)
    
    ElMessage.success('登录成功')
    showUserDialog.value = false
    
    // 重置表单
    userForm.username = ''
    userForm.email = ''
    
  } catch (error) {
    console.error('Login failed:', error)
    ElMessage.error('登录失败，请重试')
  }
}

const handleUserDialogClose = () => {
  showUserDialog.value = false
  userForm.username = ''
  userForm.email = ''
}

const logout = () => {
  currentUser.value = ''
  localStorage.removeItem('currentUser')
  localStorage.removeItem('userId')
  ElMessage.success('已退出登录')
}

// 生命周期
onMounted(() => {
  // 检查本地存储的用户信息
  const savedUser = localStorage.getItem('currentUser')
  if (savedUser) {
    currentUser.value = savedUser
  } else {
    // 如果没有登录，显示登录对话框
    showUserDialog.value = true
  }
})
</script>

<style scoped>
/* 主容器 */
.app-container {
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* 侧边栏 */
.sidebar {
  background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
  box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.sidebar::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
  opacity: 0.3;
  pointer-events: none;
}

.sidebar-header {
  padding: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  position: relative;
  z-index: 1;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #ff6b6b, #feca57);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 20px;
  box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
}

.logo-text {
  color: white;
}

.logo-title {
  font-size: 18px;
  font-weight: 700;
  line-height: 1.2;
}

.logo-subtitle {
  font-size: 12px;
  opacity: 0.8;
  margin-top: 2px;
}

.collapse-btn {
  background: rgba(255, 255, 255, 0.1) !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  color: white !important;
  transition: all 0.3s ease;
}

.collapse-btn:hover {
  background: rgba(255, 255, 255, 0.2) !important;
  transform: scale(1.05);
}

/* 侧边栏菜单 */
.sidebar-menu {
  border: none !important;
  padding: 20px 0;
  position: relative;
  z-index: 1;
}

.sidebar-menu .menu-item {
  margin: 4px 16px;
  border-radius: 12px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.sidebar-menu .menu-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
  transition: left 0.5s ease;
}

.sidebar-menu .menu-item:hover::before {
  left: 100%;
}

.sidebar-menu .menu-item:hover {
  background: rgba(255, 255, 255, 0.1) !important;
  transform: translateX(4px);
}

.sidebar-menu .menu-item.is-active {
  background: linear-gradient(135deg, #ff6b6b, #feca57) !important;
  color: white !important;
  box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
}

.sidebar-menu .menu-item .el-icon {
  font-size: 18px;
  margin-right: 12px;
}

/* 侧边栏底部 */
.sidebar-footer {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
}

.user-profile {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  padding: 8px;
  border-radius: 12px;
  transition: all 0.3s ease;
}

.user-profile:hover {
  background: rgba(255, 255, 255, 0.1);
}

.user-avatar {
  background: linear-gradient(135deg, #667eea, #764ba2);
  border: 2px solid rgba(255, 255, 255, 0.2);
}

.user-info {
  color: white;
}

.user-name {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.2;
}

.user-role {
  font-size: 12px;
  opacity: 0.8;
  margin-top: 2px;
}

/* 主容器 */
.main-container {
  background: #f8fafc;
  min-height: 100vh;
}

.main-header {
  background: white;
  border-bottom: 1px solid #e2e8f0;
  padding: 0 32px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.header-left .el-breadcrumb {
  font-size: 14px;
}

.header-right .el-button-group .el-button {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  color: #64748b;
  transition: all 0.3s ease;
}

.header-right .el-button-group .el-button:hover {
  background: #e2e8f0;
  color: #475569;
  transform: translateY(-1px);
}

.main-content {
  padding: 0;
  background: #f8fafc;
}

.content-wrapper {
  max-width: 98vw;
  margin: 0 auto;
  padding: 24px;
  min-width: 1200px;
}

/* 对话框样式 */
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

/* 响应式设计 */
@media (min-width: 1920px) {
  .content-wrapper {
    max-width: 98vw;
    padding: 32px;
  }
}

@media (min-width: 1600px) {
  .content-wrapper {
    max-width: 98vw;
    padding: 28px;
  }
}

@media (max-width: 1400px) {
  .content-wrapper {
    max-width: 95vw;
    padding: 20px;
  }
}

@media (max-width: 768px) {
  .sidebar {
    position: fixed;
    z-index: 1000;
    height: 100vh;
  }
  
  .content-wrapper {
    padding: 20px;
    min-width: auto;
    max-width: 100vw;
  }
  
  .main-header {
    padding: 0 20px;
  }
}
</style>

<style>
/* 全局样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
}

#app {
  height: 100vh;
}

.el-container {
  height: 100%;
}

/* 自定义滚动条 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
