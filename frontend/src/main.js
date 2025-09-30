import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

import App from './App.vue'
import Dashboard from './components/Dashboard.vue'
import TaskList from './components/TaskList.vue'
import TaskEditor from './components/TaskEditor.vue'
import ModelManager from './components/ModelManager.vue'

// 路由配置
const routes = [
  { path: '/', name: 'Dashboard', component: Dashboard },
  { path: '/tasks', name: 'TaskList', component: TaskList },
  { path: '/tasks/new', name: 'TaskEditor', component: TaskEditor },
  { path: '/tasks/:id/edit', name: 'TaskEdit', component: TaskEditor },
  { path: '/models', name: 'ModelManager', component: ModelManager }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 创建应用
const app = createApp(App)

// 注册 Element Plus 图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 使用插件
app.use(router)
app.use(ElementPlus, {
  locale: zhCn,
})

// 挂载应用
app.mount('#app')
