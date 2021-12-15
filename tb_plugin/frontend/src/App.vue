<template>
  <div id="app">
    <b-container>
      <b-row>
        <b-col cols="3">
          <Sidebar/>
        </b-col>
        <b-col cols="9">
          <b-card>
            <MainContent v-if="data !== null"/>
          </b-card>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import MainContent from './components/MainContent.vue'
import Sidebar from "./components/Sidebar";
import {mapState} from "vuex";

export default {
  name: 'App',
  components: {
    Sidebar,
    MainContent,
  },
  computed: {
    ...mapState({
      data: state => state.data,
    }),
  },
  mounted() {
    this.$store.dispatch('loadRuns');
  },
}
</script>

<style lang="scss">
</style>
