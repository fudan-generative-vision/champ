import HomeView from '@/views/HomeView.vue'
import { createRouter, createWebHashHistory, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '',
      component: HomeView
    }
  ]
})

export default router
