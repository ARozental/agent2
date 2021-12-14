module.exports = {
    publicPath: process.env.NODE_ENV === 'production' ? '' : '',
    devServer: {
        proxy: {
            '/tags': {target: 'http://localhost:6006/data/plugin/AGENT'},
        },
    },
}