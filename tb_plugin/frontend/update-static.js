const fs = require('fs')
const path = require('path')

DEST_DIRECTORY = 'agent_plugin'

fs.copyFileSync(
    path.resolve(__dirname, 'dist/index.html'),
    path.resolve(__dirname, `../${DEST_DIRECTORY}/static/index.html`)
)

console.log('Copy done.')