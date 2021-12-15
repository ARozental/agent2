<template>
  <b-card>
    <h6>Run</h6>
    <b-list-group style="max-height: 400px; overflow-y: scroll;">
      <b-list-group-item v-for="run in runs" :key="run"
                         href="#" class="small py-0 px-1"
                         :active="selected.run_id === run"
                         @click.stop="updateSelectedRun(run)">
        {{ run }}
      </b-list-group-item>
    </b-list-group>

    <br>
    <b-checkbox v-model="use_e">&nbsp;Use embedding "_e"?</b-checkbox>
  </b-card>
</template>

<script>
import {mapState} from "vuex";

export default {
  name: "Sidebar",
  computed: {
    ...mapState({
      runs: state => state.runs,
      selected: state => state.selected,
      data: state => state.data,
    }),
    use_e: {
      get() {
        return this.selected.use_e;
      },
      set(value) {
        this.$store.commit('setSelectedUseE', value);
      },
    },
  },
  methods: {
    updateSelectedRun(run_id) {
      this.$store.commit('setSelectedRunID', run_id);
      this.$store.dispatch('getData');
    },
  },
}
</script>

<style scoped>

</style>