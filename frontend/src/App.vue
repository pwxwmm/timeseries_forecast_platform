<template>
  <div id="app">
    <el-container>
      <!-- 顶部导航栏 -->
      <el-header class="header">
        <div class="header-content">
          <div class="logo">
            <el-icon><TrendCharts /></el-icon>
            <span>时间序列预测平台</span>
          </div>
          <el-menu
            :default-active="$route.path"
            mode="horizontal"
            router
            class="nav-menu"
          >
            <el-menu-item index="/">
              <el-icon><House /></el-icon>
              <span>仪表板</span>
            </el-menu-item>
            <el-menu-item index="/tasks">
              <el-icon><List /></el-icon>
              <span>任务管理</span>
            </el-menu-item>
            <el-menu-item index="/models">
              <el-icon><Cpu /></el-icon>
              <span>模型管理</span>
            </el-menu-item>
          </el-menu>
          <div class="user-info">
            <el-dropdown>
              <span class="user-name">
                <el-icon><User /></el-icon>
                {{ currentUser || '未登录' }}
              </span>
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
        </div>
      </el-header>

      <!-- 主内容区域 -->
      <el-main class="main-content">
        <router-view />
      </el-main>
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
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from './api'

// 响应式数据
const currentUser = ref('')
const showUserDialog = ref(false)
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

// 方法
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
.header {
  background-color: #fff;
  border-bottom: 1px solid #e4e7ed;
  padding: 0;
  height: 60px;
  line-height: 60px;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.logo {
  display: flex;
  align-items: center;
  font-size: 20px;
  font-weight: bold;
  color: #409eff;
}

.logo .el-icon {
  margin-right: 8px;
  font-size: 24px;
}

.nav-menu {
  flex: 1;
  margin: 0 40px;
  border-bottom: none;
}

.nav-menu .el-menu-item {
  height: 60px;
  line-height: 60px;
}

.user-info {
  display: flex;
  align-items: center;
}

.user-name {
  display: flex;
  align-items: center;
  cursor: pointer;
  color: #606266;
}

.user-name .el-icon {
  margin-right: 4px;
}

.main-content {
  background-color: #f5f5f5;
  min-height: calc(100vh - 60px);
  padding: 20px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
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
