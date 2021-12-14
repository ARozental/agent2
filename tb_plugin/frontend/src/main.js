import Vue from 'vue'
import App from './App.vue'
import store from './store'
import {BootstrapVue} from 'bootstrap-vue'

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

Vue.use(BootstrapVue)

import VueSlider from 'vue-slider-component'
import 'vue-slider-component/theme/default.css'

Vue.component('VueSlider', VueSlider)

Vue.config.productionTip = false

new Vue({
    store,
    render: function (h) {
        return h(App)
    }
}).$mount('#app')
