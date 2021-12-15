<template>
  <div>
    <span v-for="(part, i) in textParts" :key="i"
          class="border border-right"
          :style="bgStyle(i)"
          v-b-tooltip.hover :title="tooltipTitle(i)">
      {{ part }}&nbsp;
    </span>
  </div>
</template>

<script>
import * as d3 from "d3-scale-chromatic";

export default {
  name: "TextDisplay",
  props: ['text', 'expected', 'updateGate'],
  computed: {
    textParts() {
      return this.text.split(' ');
    },
  },
  methods: {
    bgStyle(index) {
      if (this.updateGate === null)
        return null;

      return 'background-color: ' + d3.interpolateGreens(this.updateGate[index]) + ';';
    },
    tooltipTitle(index) {
      if (this.updateGate === null)
        return null;

      return this.updateGate[index].toFixed(3);
    }
  },
}
</script>

<style scoped>

</style>