const HtmlWebpackPlugin = require('html-webpack-plugin');
const InlineChunkHtmlPlugin = require('react-dev-utils/InlineChunkHtmlPlugin');

module.exports = {
    productionSourceMap: false,
    publicPath: process.env.NODE_ENV === 'production' ? '' : '',
    devServer: {
        proxy: {
            '/runs': {target: 'http://localhost:6006/data/plugin/agent'},
            '/data': {target: 'http://localhost:6006/data/plugin/agent'},
        },
    },
    css: {
        extract: false,
    },
    configureWebpack: {
        optimization: {
            splitChunks: false // makes there only be 1 js file - leftover from earlier attempts but doesn't hurt
        },
        plugins: [
            new HtmlWebpackPlugin({
                inject: true,
                template: 'public/index.html',
            }),
            new InlineChunkHtmlPlugin(HtmlWebpackPlugin, [/\.(js|css)$/]),
        ]
    }
}