// Real paper acceptance: needs running service, DeepSeek, Playwright and Edge.
const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
(async()=>{
 const output=path.resolve(__dirname,'../results/daily-real');fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1280,height:980},recordVideo:{dir:output,size:{width:1280,height:980}}});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const list=await(await context.request.get('http://127.0.0.1:8000/documents')).json();
 const doc=list.documents.find(item=>item.filename==='RAG原始论文.pdf');assert.ok(doc,'Expected real RAG paper');
 await page.goto('http://127.0.0.1:8000');await page.getByText('本地服务已连接',{exact:false}).waitFor();
 await page.locator('#all-documents').uncheck();await page.getByRole('checkbox',{name:`查询 ${doc.filename}`,exact:true}).check();
 async function ask(question){
  await page.locator('#question').fill(question);
  const waiting=page.waitForResponse(response=>response.url().endsWith('/query')&&response.request().method()==='POST',{timeout:240000});
  await page.locator('#ask').click();const response=await waiting;
  assert.equal(response.status(),200);const data=await response.json();assert.equal(data.status,'answered');assert.ok(data.citations.length);
  assert.ok(data.results.every(item=>item.asset_id===doc.document_id),'Other document entered chosen scope');
  await page.locator('#message').filter({hasText:'查询完成'}).waitFor();return data;
 }
 const first=await ask('这篇论文中的 RAG-Sequence 和 RAG-Token 有什么区别？');
 const second=await ask('那它们对检索到的文档是如何使用的？');
 assert.ok(second.history_used);assert.ok(second.retrieval_query);
 assert.equal(await page.locator('#history .turn').count(),1);
 const citation=second.citations[0];assert.ok(citation.source_url.includes('#page='));
 const original=await context.request.get('http://127.0.0.1:8000'+citation.source_url.split('#')[0]);
 assert.equal(original.status(),200);assert.equal(original.headers()['content-type'],'application/pdf');
 assert.equal((await original.body()).subarray(0,5).toString(),'%PDF-');
 await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
 const [popup]=await Promise.all([context.waitForEvent('page'),page.locator('#citations a').first().click()]);
 await popup.waitForURL(url=>url.href.includes('/documents/')&&url.href.includes('/file'),{timeout:15000});
 const openedUrl=popup.url();assert.ok(openedUrl.includes('#page='));await popup.close();
 await page.reload();await page.getByText('本地服务已连接',{exact:false}).waitFor();
 assert.equal(await page.locator('#history .turn').count(),1);assert.equal(await page.locator('#current-question').textContent(),'那它们对检索到的文档是如何使用的？');
 assert.equal(await page.getByRole('checkbox',{name:`查询 ${doc.filename}`,exact:true}).isChecked(),true);
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});assert.deepEqual(errors,[]);
 const video=page.video();await context.close();const recording=await video.path();await browser.close();
 const report={status:'passed',asset_id:doc.document_id,filename:doc.filename,first,followup:second,pdf_url:openedUrl,reload_history_preserved:true,mobile_overflow:false,page_errors:errors,recording};
 fs.writeFileSync(path.join(output,'browser-validation.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({status:'passed',paper:doc.filename,first_citations:first.citations.length,followup_citations:second.citations.length,retrieval_query:second.retrieval_query,history_used:second.history_used,pdf_url:openedUrl,recording,errors}));
})().catch(error=>{console.error(error.stack);process.exitCode=1});
