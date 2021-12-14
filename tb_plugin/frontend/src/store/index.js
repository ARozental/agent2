import Vue from 'vue'
import Vuex from 'vuex'
import axios from "axios";

Vue.use(Vuex)

export default new Vuex.Store({
    state: {
        runs: null,
        tags: null,
        selected: {
            run_id: null,
            tag: null,
        },
        data: null,
    },
    mutations: {
        setSelectedRunID(state, run_id) {
            state.selected.run_id = run_id;
        },
        setSelectedTag(state, tag) {
            state.selected.tag = tag;
        },
        setRunsTags(state, data) {
            state.runs = data.map(run => run.id);
            state.tags = Array.from(new Set([].concat(...data.map(run => run.tags))));
        },
        setData(state, data) {
            state.data = data;
        },
    },
    actions: {
        loadRuns({commit}) {
            axios
                .get('./runs')
                .then(r => r.data)
                .then(response => commit('setRunsTags', response));
        },
        getData({commit, state}) {
            if (state.selected.run_id === null || state.selected.tag === null)
                return;

            axios
                .get('./data', {
                    params: {
                        run_id: state.selected.run_id,
                        tag: state.selected.tag
                    }
                })
                .then(r => r.data)
                .then(response => commit('setData', response));
        },
    },
    modules: {}
})
