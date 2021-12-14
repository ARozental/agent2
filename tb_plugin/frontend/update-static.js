const fs = require('fs')
const path = require('path')

DEST_DIRECTORY = 'agent_plugin'

fs.cpSync(
    path.resolve(__dirname, 'dist/css'),
    path.resolve(__dirname, `../${DEST_DIRECTORY}/static/css`),
    {recursive: true}
);

fs.cpSync(
    path.resolve(__dirname, 'dist/img'),
    path.resolve(__dirname, `../${DEST_DIRECTORY}/static/img`),
    {recursive: true}
);

fs.cpSync(
    path.resolve(__dirname, 'dist/js'),
    path.resolve(__dirname, `../${DEST_DIRECTORY}/static/js`),
    {recursive: true}
);

fs.copyFileSync(
    path.resolve(__dirname, 'dist/index.html'),
    path.resolve(__dirname, `../${DEST_DIRECTORY}/static/index.html`)
)

console.log('Copy done.')