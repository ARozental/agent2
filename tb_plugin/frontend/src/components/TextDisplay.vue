<template>
  <div>
    {{ data[data.length - 1].text }}
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
  },
}
</script>

<style scoped>

</style>