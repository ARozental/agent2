const HtmlWebpackPlugin = require('html-webpack-plugin');
const InlineChunkHtmlPlugin = require('react-dev-utils/InlineChunkHtmlPlugin');

module.exports = {
    productionSourceMap: false,
    publicPath: process.env.NODE_ENV === 'production' ? '' : '',
    devServer: {
        proxy: {
            '/tags': {target: 'http://localhost:6006/data/plugin/AGENT'},
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
                // scriptLoading: 'blocking',
                // filename: 'index.html', // the output file name that will be created
                template: 'public/index.html', // this is important - a template file to use for insertion
            }),
            new InlineChunkHtmlPlugin(HtmlWebpackPlugin, [/\.(js|css)$/]),
        ]
    }
}