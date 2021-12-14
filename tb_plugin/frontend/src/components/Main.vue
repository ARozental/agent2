<template>
  <b-container>
    <b-row>
      <b-col cols="3">
        <b-card>
          <h6>Run</h6>
          <b-select v-model="selected_run" size="sm">
            <b-select-option :value="null">Select One</b-select-option>
            <b-select-option v-for="run in runs" :key="run" :value="run">{{ run }}</b-select-option>
          </b-select>

          <h6 class="mt-4">Tag</h6>
          <b-select v-model="selected_tag" size="sm">
            <b-select-option :value="null">Select One</b-select-option>
            <b-select-option v-for="tag in tags" :key="tag" :value="tag">{{ tag }}</b-select-option>
          </b-select>
        </b-card>
      </b-col>
      <b-col cols="9">
        <b-card>
          <TextDisplay v-if="data !== null"/>
        </b-card>
      </b-col>
    </b-row>
  </b-container>
</template>

<script>
import {mapState} from "vuex";
import TextDisplay from "./TextDisplay";

export default {
  name: 'Main',
  components: {TextDisplay},
  computed: {
    ...mapState({
      runs: state => state.runs,
      tags: state => state.tags,
      selected: state => state.selected,
      data: state => state.data,
    }),
    selected_run: {
      get() {
        return this.selected.run_id;
      },
      set(value) {
        this.$store.commit('setSelectedRunID', value);
        this.$store.dispatch('getData');
      },
    },
    selected_tag: {
      get() {
        return this.selected.tag;
      },
      set(value) {
        this.$store.commit('setSelectedTag', value);
        this.$store.dispatch('getData');
      },
    }
  },
}
</script>

<style scoped lang="scss">
</style>
