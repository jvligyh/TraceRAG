// UI acceptance uses stub HTTP responses, so it does not spend API credits.
const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
(async()=>{
 const html=fs.readFileSync(path.resolve(__dirname,'../src/tracerag/web.html'),'utf8');
 new Function(html.split('<script>')[1].split('</script>')[0]);
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1280,height:900}});
 const page=await context.newPage();
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const queries=[];let count=0;
 const docs=[{document_id:'a',filename:'A.pdf',chunks:5,original_available:true},{document_id:'b',filename:'B.md',chunks:3,original_available:true}];
 await context.route('http://tracerag.test/**',async route=>{
  const request=route.request(),url=new URL(request.url());
  if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
  let data;
  if(url.pathname==='/health')data={status:'ok',provider:'deepseek',models_loaded:true};
  else if(url.pathname==='/documents'&&request.method()==='GET')data={documents:docs};
  else if(url.pathname==='/documents'&&request.method()==='POST'){count++;const item={document_id:'new'+count,filename:'upload'+count+'.md',chunks:1,original_available:true};docs.push(item);data=item}
  else if(url.pathname==='/query'){
   const payload=request.postDataJSON();queries.push(payload);
   if(payload.question==='fail')return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'模型暂不可用'})});
   data={status:'answered',answer:'有证据的回答'+queries.length,results:[],citations:[{chunk_id:'c1',source:'A.pdf',page_number:2,text:'Actual evidence',source_url:'/documents/a/file#page=2',original_available:true}]};
  }else return route.fulfill({status:404,body:'not found'});
  return route.fulfill({contentType:'application/json',body:JSON.stringify(data)});
 });
 await page.goto('http://tracerag.test');
 await page.getByText('本地服务已连接',{exact:false}).waitFor();
 await page.locator('#all-documents').uncheck();
 await page.getByRole('checkbox',{name:'查询 A.pdf',exact:true}).check();
 async function ask(question){await page.locator('#question').fill(question);await page.locator('#ask').click();await page.locator('#message').filter({hasText:'查询完成'}).waitFor()}
 await ask('第一问');assert.deepEqual(queries[0].document_ids,['a']);assert.equal(queries[0].history.length,0);
 assert.equal(await page.locator('#citations a').getAttribute('href'),'/documents/a/file#page=2');
 await ask('它有什么限制？');assert.equal(queries[1].history.length,2);assert.equal(queries[1].history[0].content,'第一问');
 assert.equal(await page.locator('#history .turn').count(),1);
 await page.reload();await page.getByText('本地服务已连接',{exact:false}).waitFor();
 assert.equal(await page.locator('#history .turn').count(),1);assert.equal(await page.locator('#all-documents').isChecked(),false);
 await page.getByRole('checkbox',{name:'查询 B.md',exact:true}).check();assert.equal(await page.locator('#latest-turn').isVisible(),false);
 await page.locator('#file').setInputFiles([{name:'one.md',mimeType:'text/markdown',buffer:Buffer.from('one')},{name:'two.md',mimeType:'text/markdown',buffer:Buffer.from('two')}]);
 await page.locator('#upload').click();await page.locator('#upload-status').filter({hasText:'成功 2/2'}).waitFor();
 assert.equal(await page.getByRole('checkbox',{name:'查询 upload1.md',exact:true}).isChecked(),true);
 assert.equal(await page.getByRole('checkbox',{name:'查询 A.pdf',exact:true}).isChecked(),false);
 await page.locator('#question').fill('fail');await page.locator('#ask').click();await page.locator('#message').filter({hasText:'输入已保留'}).waitFor();assert.equal(await page.locator('#question').inputValue(),'fail');
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 assert.deepEqual(errors,[]);await browser.close();console.log(JSON.stringify({status:'passed',cases:['scope isolation','follow-up history','reload persistence','scope reset','multi-upload selection','source link','error retry','mobile overflow'],errors}));
})().catch(error=>{console.error(error.stack);process.exitCode=1});
