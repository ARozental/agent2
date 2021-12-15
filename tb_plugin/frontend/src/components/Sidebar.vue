<template>
  <b-card>
    <h6>Run</h6>
    <b-select v-model="selected_run" size="sm">
      <b-select-option :value="null">Select One</b-select-option>
      <b-select-option v-for="run in runs" :key="run" :value="run">{{ run }}</b-select-option>
    </b-select>

    <br>
    <b-checkbox>Using embedding?</b-checkbox>
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
      selected_run_id: state => state.selected_run_id,
      data: state => state.data,
    }),
    selected_run: {
      get() {
        return this.selected_run_id;
      },
      set(value) {
        this.$store.commit('setSelectedRunID', value);
        this.$store.dispatch('getData');
      },
    },
  },
}
</script>

<style scoped>

</style>