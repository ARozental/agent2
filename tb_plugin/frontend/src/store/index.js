import Vue from 'vue'
import Vuex from 'vuex'
import axios from "axios";

Vue.use(Vuex)

export default new Vuex.Store({
    state: {
        runs: null,
        tags: null,
        selected_run_id: null,
        data: null,
    },
    mutations: {
        setSelectedRunID(state, run_id) {
            state.selected_run_id = run_id;
        },
        setRunsTags(state, data) {
            state.runs = data.map(run => run.id);
            state.tags = ['reconstructed/1/text_summary'] //Array.from(new Set([].concat(...data.map(run => run.tags))));
        },
        setData(state, data) {
            console.log(data);
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
            if (state.selected_run_id === null)
                return;

            state.data = null;
            axios
                .get('./data', {
                    params: {
                        run_id: state.selected_run_id,
                    }
                })
                .then(r => r.data)
                .then(response => commit('setData', response));
        },
    },
    modules: {}
})
