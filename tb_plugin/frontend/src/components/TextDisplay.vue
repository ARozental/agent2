<template>
  <div>
    <b-jumbotron>
      {{ text }}
    </b-jumbotron>
    <br>
    <vue-slider v-model="currentStep" :marks="steps" :adsorb="true" :included="true" :max="maxSteps"></vue-slider>
  </div>
</template>

<script>
import {mapState} from "vuex";

export default {
  name: "TextDisplay",
  computed: {
    ...mapState({
      data: state => state.data,
    }),
    steps() {
      return this.data.map(item => item.step);
    },
    maxSteps() {
      return this.steps[this.steps.length - 1];
    },
    text() {
      let item = this.data.find(x => x.step === this.currentStep);
      if (item === undefined) {
        return 'Unable to find';
      }

      return item.text;
    },
  },
  data() {
    return {
      currentStep: 0,
      marks2: [0, 10, 30, 90, 100],
    }
  },
  mounted() {
    this.currentStep = this.data[this.data.length - 1].step;
  },
}
</script>

<style scoped>

</style>