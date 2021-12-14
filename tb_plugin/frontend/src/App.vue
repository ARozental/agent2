<template>
  <div id="app">
    <HelloWorld msg="Welcome to Your Vue.js App"/>
  </div>
</template>

<script>
import HelloWorld from './components/HelloWorld.vue'
import axios from 'axios';

export default {
  name: 'App',
  components: {
    HelloWorld
  },
  mounted() {
    axios
        .get('./runs')
        .then(r => r.data)
        .then(response => {
          console.log('runs', response)

          let name = Object.keys(response)[0];

          axios
              .get('./data', {
                params: {
                  run_id: name,
                  tag: response[name][0],
                }
              })
              .then(r => r.data)
              .then(response => console.log('data', response));
        });
  }
}
</script>

<style lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
