<template>
  <b-card>
    <h6>Run</h6>
    <b-select v-model="run_id" size="sm">
      <b-select-option :value="null">Select One</b-select-option>
      <b-select-option v-for="run in runs" :key="run" :value="run">{{ run }}</b-select-option>
    </b-select>

    <br>
    <b-checkbox v-model="use_e">Using embedding?</b-checkbox>
  </b-card>
</template>

<script>
import {mapState} from "vuex";

export default {
  name: "Sidebar",
  computed: {
    ...mapState({
      runs: state => state.runs,
      tags: state => state.tags,
      selected: state => state.selected,
      data: state => state.data,
    }),
    run_id: {
      get() {
        return this.selected.run_id;
      },
      set(value) {
        this.$store.commit('setSelectedRunID', value);
        this.$store.dispatch('getData');
      },
    },
    use_e: {
      get() {
        return this.selected.use_e;
      },
      set(value) {
        this.$store.commit('setSelectedUseE', value);
      },
    },
  },
}
</script>

<style scoped>

</style>