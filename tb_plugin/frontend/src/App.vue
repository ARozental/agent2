<template>
  <div id="app">
    <Main/>
  </div>
</template>

<script>
import Main from './components/Main.vue'
import axios from 'axios';

export default {
  name: 'App',
  components: {
    Main
  },
  mounted() {
    this.$store.dispatch('loadRuns');
  }
}
</script>

<style lang="scss">
</style>
