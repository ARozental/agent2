<template>
  <div v-if="item !== undefined">
    <b-card v-for="level in [0, 1]" class="mb-4">
      <b-card-title>Level {{ level }}</b-card-title>
      <TextDisplay :text="item['reconstructed']['reconstructed' + useE + '/' + level + '/text_summary']"
                   :expected="item.expected"
                   :update-gate="item.pndb.update_gate"/>
    </b-card>
    <br>
    <vue-slider v-model="currentStep" :marks="steps" :adsorb="true" :included="true" :max="maxSteps"></vue-slider>
  </div>
</template>

<script>
import {mapState} from "vuex";
import TextDisplay from "./TextDisplay";

export default {
  name: "MainContent",
  components: {TextDisplay},
  computed: {
    ...mapState({
      data: state => state.data,
      use_e: state => state.selected.use_e,
    }),
    steps() {
      return this.data.map(item => item.step);
    },
    maxSteps() {
      return this.steps[this.steps.length - 1];
    },
    item() {
      return this.data.find(x => x.step === this.currentStep);
    },
    useE() {
      return this.use_e ? '_e' : '';
    }
  },
  data() {
    return {
      currentStep: 0,
    }
  },
  mounted() {
    this.currentStep = this.data[this.data.length - 1].step;
  },
}
</script>

<style scoped>

</style>